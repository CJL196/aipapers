# 1. Bibliographic Information

## 1.1. Title
MemFlow: Flowing Adaptive Memory for Consistent and Efficient Long Video Narratives

The title clearly indicates the paper's focus: a method named `MemFlow` designed to improve **long video generation**. The key challenges it addresses are **consistency** (maintaining a coherent narrative and appearance) and **efficiency** (generating videos quickly). The core idea is a "flowing adaptive memory," which suggests a dynamic memory system that changes over time.

## 1.2. Authors
The authors are Sihui Ji, Xi Chen, Shuai Yang, Xin Tao, Pengfei Wan, Hengshuang Zhao. Their affiliations are:
*   **The University of Hong Kong (HKU)**: A leading research university.
*   **Kuaishou Technology (Kling Team)**: A major Chinese tech company known for its short-video platform. The Kling Team is their research division focused on video generation, indicating a strong industry connection and application-driven focus.
*   **The Hong Kong University of Science and Technology (Guangzhou) (HKUST(GZ))**: Another top-tier research university.

    The collaboration between top academic institutions and a leading industry player in video technology suggests that the research is both academically rigorous and practically relevant.

## 1.3. Journal/Conference
The paper provides a publication date of December 16, 2025, and an arXiv link. This indicates it is a **preprint**, meaning it has been made publicly available before or during a peer-review process for a conference or journal. The date suggests a submission to a future top-tier conference in computer vision (like CVPR) or machine learning (like NeurIPS, ICLR). The venue is not yet determined.

## 1.4. Publication Year
The publication year on arXiv is listed as 2025.

## 1.5. Abstract
The abstract summarizes the paper's core problem, proposed solution, and key results.
*   **Problem:** The main difficulty in streaming video generation is maintaining **content consistency** over long durations. Existing methods use fixed memory compression strategies, which are not ideal because different parts of a video may require different historical information (cues).
*   **Methodology:** The paper proposes `MemFlow`, which uses a dynamic memory bank. Before generating a new video "chunk," `MemFlow` retrieves the most relevant historical frames based on the text prompt for that chunk. This ensures both visual and narrative coherence. To maintain efficiency, it only activates the most relevant parts ("tokens") of the memory during the generation process.
*   **Results:** `MemFlow` achieves excellent long-term consistency with a very small computational cost (only a 7.9% speed reduction compared to a baseline without memory). It is also compatible with existing streaming video generation models that use a `KV cache`.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2512.14699
*   **PDF Link:** https://arxiv.org/pdf/2512.14699v1.pdf
*   **Status:** This is a preprint on arXiv and has not yet been officially published in a peer-reviewed venue.

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
*   **Core Problem:** Generating high-quality, long videos (e.g., several minutes) that are internally consistent is a major challenge in AI. While models can create impressive short clips, they struggle to maintain the appearance of characters, objects, and scenes over extended durations. When generating a video piece-by-piece (in "chunks"), the model can easily "forget" what happened earlier, leading to inconsistencies like a character suddenly changing clothes or a background object disappearing.

*   **Specific Challenges:**
    1.  **Memory Management:** To be consistent, a model needs to "remember" previous frames. Storing all historical frames is computationally impossible due to GPU memory limitations. Existing methods compress history, but they often use fixed, predefined strategies (e.g., always keeping the first frame, or compressing the last few frames).
    2.  **Dynamic Context Needs:** A fixed memory strategy is suboptimal. For example, if a prompt says "a cat is now chasing the dog," the model needs to recall what the "dog" looked like from earlier frames. If the prompt changes to "the scene shifts to a quiet library," recalling the dog is irrelevant and could even be harmful. The memory needs to be **adaptive** to the current narrative context provided by the text prompt.
    3.  **Efficiency:** Adding a memory mechanism inevitably slows down the generation process. An ideal solution must be efficient enough for practical, near-real-time applications like interactive video creation.

*   **Paper's Entry Point:** The authors' innovative idea is to create a "flowing adaptive memory." Instead of a static or rigidly updated memory, `MemFlow` dynamically selects which historical information is most relevant for the *upcoming* video chunk by comparing the history to the new text prompt. This ensures the memory is always tailored to the current narrative need. Furthermore, it intelligently prunes the memory during computation to keep the process fast.

## 2.2. Main Contributions / Findings
*   **Narrative Adaptive Memory (NAM):** This is the core contribution. NAM is a dynamic memory management mechanism.
    *   It maintains a memory bank of historical visual information.
    *   Before generating a new video chunk, it uses the **text prompt** for that chunk to perform a **semantic retrieval**, finding and prioritizing historical frames that are most relevant to the prompt's content.
    *   This allows the model to maintain consistency for existing elements and coherently handle the introduction of new elements or scene changes.

*   **Sparse Memory Activation (SMA):** This mechanism addresses the efficiency problem.
    *   Instead of using the entire (retrieved) memory for computation, SMA identifies and activates only the most relevant memory tokens for each part of the new frame being generated.
    *   This is achieved by a `top-k` selection based on relevance scores, drastically reducing the computational load of the attention mechanism.

*   **State-of-the-Art Performance:** The paper reports that `MemFlow` achieves top results in long video generation, particularly for interactive, multi-prompt scenarios. It demonstrates superior consistency and adherence to narrative prompts while being highly efficient, with only a 7.9% speed reduction compared to a memory-free baseline and achieving 18.7 FPS on a high-end GPU. The method is also a plug-and-play module compatible with many existing video generation frameworks.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that have become state-of-the-art for generating high-quality images and videos. They work in two stages:
1.  **Forward Process (Noise Addition):** Starting with a real image or video frame, a small amount of Gaussian noise is added iteratively over many steps. By the end, the original data is transformed into pure, random noise. This process is fixed and doesn't involve any learning.
2.  **Reverse Process (Denoising):** This is where the magic happens. A neural network (often a U-Net or a Transformer) is trained to reverse the process: given a noisy image, it predicts the noise that was added. By repeatedly subtracting the predicted noise, the model can generate a clean, realistic image starting from pure random noise. To guide the generation (e.g., with a text prompt), this conditioning information is fed into the denoising network at each step.

### 3.1.2. Transformer Architecture and Attention Mechanism
The Transformer is a neural network architecture that excels at handling sequential data. It was originally designed for natural language processing but is now widely used in vision, including in models like the `Diffusion Transformer (DiT)` mentioned in the paper.
*   **Self-Attention:** The core component of a Transformer is the **self-attention mechanism**. For each element (e.g., a word in a sentence or a patch in an image), self-attention calculates an "attention score" that determines how much focus to place on all other elements in the sequence when processing the current one. This allows the model to capture complex, long-range dependencies.
*   **Query, Key, Value (Q, K, V):** To compute attention, each input element is projected into three vectors: a **Query (Q)**, a **Key (K)**, and a **Value (V)**.
    *   The **Query** represents the current element's request for information.
    *   The **Key** represents what information each element in the sequence has to offer.
    *   The **Value** represents the actual content of each element.
        The attention score is calculated by taking the dot product of a Query with all Keys. These scores are then normalized (using `softmax`) and used to create a weighted sum of all Values. The canonical formula is:
    \$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    \$
    *   $Q$, $K$, $V$ are matrices containing the query, key, and value vectors for all elements.
    *   $d_k$ is the dimension of the key vectors, used for scaling.

### 3.1.3. KV Cache
In **autoregressive generation** (generating content step-by-step), the attention mechanism can be very inefficient. When generating the 10th element, the model needs to attend to the previous 9. When generating the 11th, it attends to the previous 10. Re-computing the Key (K) and Value (V) vectors for the first 9 elements is redundant.

The **KV Cache** is an optimization that stores the K and V vectors of previously generated elements. When generating a new element, the model only needs to compute the Q vector for the new element and can reuse the cached K and V vectors from all previous steps. This dramatically speeds up inference. `MemFlow` leverages this concept by treating the `KV cache` of historical frames as its memory bank.

### 3.1.4. Autoregressive (AR) Video Generation
Autoregressive models generate sequences one element at a time, where each new element is conditioned on the ones generated before it. For video, this means generating a video "chunk" (a few frames) based on the previously generated chunk. This "chunk-wise" approach breaks down the impossibly large task of generating a long video at once into a series of smaller, manageable steps. However, it risks **error accumulation**, where small mistakes in early chunks propagate and get worse over time, leading to a loss of quality and consistency.

## 3.2. Previous Works
The paper categorizes previous long video generation methods into three groups:

1.  **Autoregressive-Diffusion Hybrid Approaches:** These are the most relevant to `MemFlow`. They generate long videos by iteratively predicting chunks of frames.
    *   `SkyReels-V2` and `MAGI-1` are large-scale models that use this paradigm.
    *   `CausVid` and `Self Forcing` focus on improving the efficiency and reducing the train-test gap in AR models. `Self Forcing` is a key technique used as a baseline and part of `MemFlow`'s training, where the model learns from its own generated outputs during training to better handle error accumulation during inference.
    *   The limitation of many of these is that they only look at a small, local window of recent frames, lacking a true long-term memory.

2.  **Multistage Methods:** These methods first generate a sequence of keyframes (like a storyboard) and then "infill" the video between them (`FramePack`, `Captain Cinema`), or they generate a sequence of prompts and use a standard text-to-video model for each segment (`MovieDreamer`). Their main weakness is often a lack of smooth temporal coherence between the separately generated clips.

3.  **Efficient Architectures:** These methods prioritize speed by using simplified attention mechanisms (e.g., linear attention) or by representing video clips with condensed "tokens" (`TokensGen`). A notable work is `Mixture of Contexts`, which dynamically selects relevant context, similar in spirit to `MemFlow`, but `MemFlow` innovates by using the text prompt for retrieval.

## 3.3. Technological Evolution
The field has evolved from generating single images to short video clips, and now to the frontier of long-form video.
1.  **Early Stage (Short Clips):** Models like `Sora` and `Kling` demonstrated high-quality video generation for a few seconds, typically using `Diffusion Transformer (DiT)` models that process the entire video at once. This is computationally expensive and not scalable to long durations.
2.  **Mid Stage (Scalability via Autoregression):** To extend duration, the field adopted autoregressive, chunk-by-chunk generation. This solved the scalability problem but introduced the consistency problem. Early AR models (`Self Forcing`) used a small sliding window of context.
3.  **Current Stage (Explicit Memory):** Researchers realized a simple sliding window is not enough. This led to the development of explicit memory mechanisms.
    *   Initial ideas were simple: `LongLive` proposed keeping the very first chunk as a permanent memory "sink."
    *   Slightly more advanced methods like `FramePack` used fixed compression schemes to store more history.
    *   `MemFlow` represents the next step in this evolution: making the memory **adaptive and dynamic**, guided by the narrative itself (the text prompts).

## 3.4. Differentiation Analysis
*   **vs. Fixed Memory (e.g., `LongLive`, `FramePack`):** `LongLive`'s "frame sink" (always remembering the first chunk) is too rigid. It can't adapt if new characters or scenes are introduced later. `FramePack`'s compression is also predefined and not context-aware. **`MemFlow`'s key innovation is its `Narrative Adaptive Memory (NAM)`, which uses the text prompt to dynamically retrieve the most relevant history, making it far more flexible for evolving narratives.**
*   **vs. Implicit Context (e.g., `Self Forcing`):** `Self Forcing` and similar methods only use a local context window (the last $n$ frames). They have no explicit long-term memory. **`MemFlow` adds an explicit, global memory bank to the local context window.**
*   **vs. Efficient Architectures (e.g., `Mixture of Contexts`):** While `Mixture of Contexts` also selects relevant context to improve efficiency, `MemFlow`'s `Sparse Memory Activation (SMA)` is specifically designed for its memory bank. More importantly, `MemFlow`'s **retrieval** mechanism (`NAM`) is guided by text prompts, which is a novel approach for ensuring semantic coherence, whereas other methods often rely on visual similarity alone. `MemFlow`'s dual system of prompt-based retrieval (`NAM`) and query-based activation (`SMA`) provides a more comprehensive solution to the consistency-efficiency trade-off.

    ---

# 4. Methodology

## 4.1. Principles
The core principle of `MemFlow` is to equip a standard autoregressive video generation model with a **smart memory system**. This system should be:
1.  **Narratively Coherent:** The memory it provides should be relevant to the story being told in the current video chunk. The best guide for this is the user's text prompt.
2.  **Efficient:** Accessing this memory should not significantly slow down the generation process.

    To achieve this, `MemFlow` introduces two main components: `Narrative Adaptive Memory (NAM)` for intelligent memory updating and `Sparse Memory Activation (SMA)` for efficient memory usage. These are integrated into a streaming, chunk-wise generation framework.

The overall architecture is shown in the figure below (Figure 2 from the paper), which illustrates how `MemFlow` integrates into an AR-diffusion model.

![该图像是示意图，展示了MemFlow的叙事自适应记忆机制。通过动态更新记忆库以检索与当前提示相关的历史帧，该方法实现了高效的记忆利用与生成效率。此外，图中显示了自回归生成模型中的记忆选择与更新过程。](images/2.jpg)
*该图像是示意图，展示了MemFlow的叙事自适应记忆机制。通过动态更新记忆库以检索与当前提示相关的历史帧，该方法实现了高效的记忆利用与生成效率。此外，图中显示了自回归生成模型中的记忆选择与更新过程。*

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Overall Framework

**Baseline Model:**
`MemFlow` builds on a standard autoregressive diffusion framework. The process works as follows:
*   A long video is generated in sequential chunks. Let's say each chunk has $T$ frames.
*   To generate the current chunk, the model is conditioned on the last $n$ frames of the *previous* chunk. This provides short-term context.
*   During generation, the Transformer-based denoising network produces a `KV cache` for all frames it processes. `MemFlow` uses this `KV cache` as the raw material for its memory bank, which is a clever way to store history without extra computation.
*   In a standard model, attention is computed over $n+T$ frames (local context + current chunk). `MemFlow` extends this by adding a memory bank of $B$ selected historical frames, so the attention now covers $n+B+T$ frames, combining short-term, long-term, and current context.

**Training Mechanism:**
The model is trained using a **streaming long-tuning** strategy based on `Self-Forcing`.
1.  A powerful, pre-trained bidirectional video generation model acts as a "teacher."
2.  The `MemFlow` model (the "student") starts generating a video chunk by chunk.
3.  After each new 5-second chunk is generated autoregressively, the teacher model provides supervision for that chunk. The student is trained to match the output distribution of the teacher using `Distribution Matching Distillation (DMD)` loss.
4.  This process is repeated, extending the video up to a maximum length (e.g., 60 seconds).
    Crucially, the `NAM` and `SMA` mechanisms are active **during this training process**. This teaches the model how to effectively retrieve and use its memory from self-generated content, aligning the training procedure with how the model will be used during inference.

### 4.2.2. Narrative Adaptive Memory (NAM)
NAM is responsible for maintaining a compact yet highly relevant memory bank. At each generation step (for a new chunk), the memory is updated. This update process involves two key strategies: **Semantic Retrieval** and **Redundant Removal**.

**1. Semantic Retrieval:**
This is the heart of `MemFlow`'s adaptiveness. Before generating the *next* chunk, the model decides which frames from its entire history are most important to keep in memory. It makes this decision by comparing the historical frames to the **text prompt of the upcoming chunk**.

*   **Process:**
    *   Let the memory bank contain the `KV caches` from $b$ historical frames.
    *   The model takes the textual embedding of the *new* prompt, which acts as a query, $Q_{\text{text}}^l$, at each transformer layer $l$.
    *   For each historical frame $i$ in the memory bank, it uses the corresponding Key cache, $K_{m,i}^l$.
    *   It calculates a semantic relevance score $S_{m,i}^l$ by performing cross-attention between the text query and the visual keys.

*   **Formula:** The relevance score for the $i$-th historical frame in the memory bank at generation iteration $m$ and layer $l$ is calculated as:
    $$
    \mathcal { S } _ { m , i } ^ { l } = \mathrm { A g g r e g a t e } \left( \mathrm { S o f t m a x } \left( \frac { Q _ { \mathrm { t e x t } } ^ { l } ( K _ { m , i } ^ { l } ) ^ { \top } } { \sqrt { d } } \right) \right)
    $$
    *   $Q_{\text{text}}^l \in \mathbb{R}^d$: The query vector from the new text prompt at layer $l$.
    *   $K_{m,i}^l \in \mathbb{R}^{n \times d}$: The key matrix for the $i$-th historical frame in memory (which has $n$ tokens).
    *   $\frac{Q_{\text{text}}^l (K_{m,i}^l)^\top}{\sqrt{d}}$: This is the standard scaled dot-product attention score calculation. It measures how much each token in the historical frame's visual content aligns with the text prompt.
    *   $\mathrm{Softmax}(\cdot)$: Normalizes these scores into attention weights.
    *   $\mathrm{Aggregate}(\cdot)$: The paper uses mean pooling. This averages the attention weights across all tokens in the frame to produce a single scalar score, $S_{m,i}^l$, representing the overall relevance of that historical frame to the new prompt.

*   **Action:** The model then selects the `top-k` historical frames with the highest relevance scores to keep in the memory bank for the next generation step.

**2. Redundant Removal:**
After retrieving the best historical frames, the model needs to add information from the chunk it *just* generated. Storing all frames from this new chunk would quickly bloat the memory.
*   **Heuristic:** The paper leverages the high temporal redundancy in short video chunks (frames right next to each other are very similar). It proposes a simple and efficient heuristic: **only the `KV cache` of the very first frame of the just-generated chunk is selected** to represent the entire chunk. This "prototype" is then added to the memory bank.

*   **Final Update:** The new memory bank is formed by concatenating the `top-k` semantically retrieved historical frames and the new prototype frame from the preceding chunk. This keeps the memory size constant while ensuring it is both up-to-date and semantically relevant to the ongoing narrative.

### 4.2.3. Sparse Memory Activation (SMA)
NAM creates a relevant memory bank, but using this entire memory in every attention layer is still computationally expensive. SMA is a technique to reduce this cost during the generation of the current chunk.

*   **Principle:** For a given query token from the current chunk being generated, it's unnecessary to attend to all tokens from all frames in the memory bank. SMA dynamically selects a small subset of the *most relevant* memory frames for each query to attend to.

*   **Process:**
    1.  **Create Descriptors:** To efficiently find relevance, the model first computes compact descriptors.
        *   For the query tokens $Q_{\text{vis}}^l$ from the current chunk, it computes a single query descriptor $\bar{q}_{\text{vis}}$ by mean pooling across the token dimension.
        *   For each of the $b$ frames in the memory bank, it computes a frame-wise key descriptor $\bar{k}_j$ by mean pooling the key tokens $K_j$.
    2.  **Calculate Relevance:** The relevance $s_j$ between the current query and the $j$-th frame in memory is calculated as a simple inner product:
        $$
        s _ { j } = \bar { q } _ { \mathrm { v i s } } ^ { \top } \bar { k } _ { j } , \quad \mathrm { f o r } \quad j = 1 , \ldots , b
        $$
        *   $\bar{q}_{\text{vis}} \in \mathbb{R}^{1 \times d}$: The single descriptor for the current query.
        *   $\bar{k}_j \in \mathbb{R}^{1 \times d}$: The single descriptor for the $j$-th memory frame.
        *   $s_j$: A scalar score indicating how relevant memory frame $j$ is to the current query.
    3.  **Select Top-k Frames:** Based on these scores, the model identifies the indices of the `top-k` most relevant memory frames.
        $$
        \mathcal { T } _ { k } = \underset { I \subseteq \{ 1 , \dots , b \} , | I | = k } { \arg \operatorname* { m a x } } \sum _ { j \in I } s _ { j }
        $$
        *   This formula simply states that we choose the set of $k$ indices ($\mathcal{T}_k$) that corresponds to the $k$ highest relevance scores $s_j$.
    4.  **Sparse Attention:** The final attention computation is then restricted to only the keys and values from these selected `top-k` frames.
        $$
        \mathrm { A t t n } ( Q _ { \mathrm { v i s } } ^ { l } , K _ { m } ^ { l } , V _ { m } ^ { l } ) \approx \mathrm { A t t n } ( Q _ { \mathrm { v i s } } ^ { l } , K _ { m , \mathcal { T } _ { k } } ^ { l } , V _ { m , \mathcal { T } _ { k } } ^ { l } )
        $$
        *   $K_{m, \mathcal{T}_k}^l$ and $V_{m, \mathcal{T}_k}^l$: The concatenated key and value tensors from *only* the `top-k` selected memory frames.

            By doing this, SMA prunes the context for the attention mechanism on-the-fly, significantly reducing computation while preserving the most critical historical information, which also helps mitigate error accumulation from irrelevant or noisy history.

---

# 5. Experimental Setup

## 5.1. Datasets
*   **Training Dataset:** The model was trained using the `switch-prompt` dataset from the `LongLive` paper. This dataset is specifically designed for long, narrative videos with changing prompts. It was constructed using the `Qwen2-72B-Instruct` language model to generate narrative scripts.
*   **Data Example:** A data sample would be a sequence of prompts describing a continuous story. For example:
    1.  (0-10s): "A golden retriever is playing fetch in a sunny park."
    2.  (10-20s): "A small child runs up and starts petting the dog."
    3.  (20-30s): "The dog licks the child's face, and they both laugh."
        This type of data is crucial for training and evaluating a model's ability to handle narrative transitions and maintain subject consistency.
*   **Evaluation Dataset:** For evaluation, the authors created a custom set of 100 narrative scripts, each consisting of 6 successive 10-second prompts, to generate 100 videos of 60 seconds each. For single-prompt generation, they used the official prompt set from the `VBench` benchmark.

## 5.2. Evaluation Metrics
The paper uses several metrics to assess performance from different angles.

### 5.2.1. VBench-Long Scores
These metrics are from the `VBench-Long` benchmark suite, designed for evaluating long video generation.
*   **Quality Score:**
    *   **Conceptual Definition:** Measures the perceptual quality of the video, including aspects like clarity, realism, and absence of artifacts.
    *   **Formula/Calculation:** This is a composite score derived from multiple automated tests within the VBench suite that assess properties like temporal flickering, motion smoothness, and visual artifacts. A specific formula is not a single equation but an aggregation of these sub-scores.

*   **Consistency Score:**
    *   **Conceptual Definition:** This is a crucial metric for this paper. It measures how well the identity and appearance of subjects and backgrounds are maintained throughout the video. A high score means characters don't change their appearance randomly and scenes remain stable unless a change is prompted.
    *   **Formula/Calculation:** This score is often calculated by measuring the feature similarity (e.g., using CLIP or DINO embeddings) of the same object or person identified across different frames of the video.

*   **Aesthetic Score:**
    *   **Conceptual Definition:** Quantifies the visual appeal of the generated video, similar to how a human might rate a photograph for its beauty or composition.
    *   **Formula/Calculation:** This is typically predicted by a pre-trained model that has been fine-tuned on human aesthetic ratings of images/videos.

### 5.2.2. CLIP Score
*   **Conceptual Definition:** Measures the semantic alignment between a given text prompt and a generated image or video frame. A higher CLIP score indicates that the content of the video frame accurately reflects the description in the text prompt.
*   **Mathematical Formula:** The score is the cosine similarity between the text and image embeddings produced by the pre-trained CLIP model.
    \$
    \text{CLIP Score} = 100 \times \cos(E_T, E_I)
    \$
*   **Symbol Explanation:**
    *   $E_T$: The feature vector (embedding) of the text prompt, obtained from the CLIP text encoder.
    *   $E_I$: The feature vector (embedding) of the video frame, obtained from the CLIP image encoder.
    *   $\cos(\cdot, \cdot)$: The cosine similarity function, which measures the angle between two vectors. A value of 1 means they are identical in orientation, 0 means they are orthogonal, and -1 means they are opposite. The score is often scaled by 100.

### 5.2.3. Throughput (FPS)
*   **Conceptual Definition:** Measures the generation speed of the model. It stands for Frames Per Second.
*   **Mathematical Formula:**
    \$
    \text{FPS} = \frac{\text{Total number of frames generated}}{\text{Total time taken in seconds}}
    \$
*   **Symbol Explanation:** This is a direct measure of inference efficiency. A higher FPS is better, especially for real-time applications.

## 5.3. Baselines
The paper compares `MemFlow` against several representative long video generation models:
*   **`SkyReels-V2` [3]:** A strong autoregressive-diffusion model.
*   **`Self Forcing` [15]:** A foundational autoregressive method. `MemFlow` uses its training paradigm.
*   **`LongLive` [35]:** The direct predecessor and primary baseline. It uses a simple "frame sink" memory (keeping the first chunk). `MemFlow` is built upon the `LongLive` framework, making this a very direct comparison of the memory mechanisms.
*   **`FramePack` [42]:** A method that uses fixed context compression for memory.
*   Other models for short video comparison include `LTX-Video`, `Wan2.1`, `MAGI-1`, `CausVid`, etc., representing a wide range of recent SOTA models.

    These baselines are well-chosen as they represent the main competing strategies for long video generation: no long-term memory (`Self Forcing`), simple fixed memory (`LongLive`), and fixed compression memory (`FramePack`).

---

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Multi-prompt Generation (60-second interactive videos)
This is the main experiment designed to test `MemFlow`'s core strength in handling long, evolving narratives. The results are presented in Table 1.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Quality Score ↑</th>
<th rowspan="2">Consistency Score ↑</th>
<th rowspan="2">Aesthetic Score ↑</th>
<th colspan="6">CLIP Score ↑</th>
</tr>
<tr>
<th>0-10 s</th>
<th>10-20 s</th>
<th>20-30 s</th>
<th>30-40 s</th>
<th>40-50 s</th>
<th>50-60 s</th>
</tr>
</thead>
<tbody>
<tr>
<td>SkyReels-V2 [3]</td>
<td>81.55</td>
<td>94.72</td>
<td>56.83</td>
<td>25.31</td>
<td>23.40</td>
<td>22.50</td>
<td>21.62</td>
<td>21.67</td>
<td>20.91</td>
</tr>
<tr>
<td>Self Forcing [15]</td>
<td>83.94</td>
<td>95.74</td>
<td>58.45</td>
<td>26.24</td>
<td>24.87</td>
<td>23.46</td>
<td>21.92</td>
<td>22.05</td>
<td>21.07</td>
</tr>
<tr>
<td>LongLive [35]</td>
<td>84.28</td>
<td>96.05</td>
<td>59.89</td>
<td>26.63</td>
<td>25.77</td>
<td>24.65</td>
<td>23.99</td>
<td>24.52</td>
<td>24.11</td>
</tr>
<tr>
<td>FramePack [15]</td>
<td>84.40</td>
<td>96.77</td>
<td>59.44</td>
<td>26.51</td>
<td>22.60</td>
<td>22.18</td>
<td>21.53</td>
<td>21.98</td>
<td>21.62</td>
</tr>
<tr>
<td>MEMFLOW</td>
<td><b>85.02</b></td>
<td>96.60</td>
<td><b>61.07</b></td>
<td><b>26.31</b></td>
<td>24.70</td>
<td>23.94</td>
<td>24.13</td>
<td><b>24.90</b></td>
<td><b>24.22</b></td>
</tr>
</tbody>
</table>

*   **Overall Quality:** `MemFlow` achieves the highest `Quality Score` (85.02) and `Aesthetic Score` (61.07), indicating that its memory mechanism not only helps with consistency but also mitigates the error accumulation that degrades visual quality over long rollouts.
*   **Consistency:** `MemFlow` (96.60) is second only to `FramePack` (96.77). The paper notes that `FramePack` tends to generate videos with less motion, which artificially inflates its consistency score. `MemFlow`'s high score demonstrates superior performance in maintaining global consistency.
*   **Narrative Coherence (CLIP Score):** The `CLIP Score` analysis is very telling. While most models show a steady decline in prompt adherence as the video gets longer (error accumulation), `MemFlow`'s scores remain strong and even increase in the later segments (e.g., 40-50s and 50-60s). This strongly supports the claim that its narrative-adaptive memory helps it stay on track with the story, even after multiple prompt switches.

    The qualitative results in Figure 3 visually confirm these numbers. `MemFlow` correctly maintains the appearance of "a woman in a casual sweater" across prompt changes, while baselines like `LongLive` introduce new, inconsistent characters.

    ![该图像是示意图，展示了不同视频生成模型在不同时间点的输出效果，包括SkyReels-V2、FramePack、Self-Forcing、LongLive和Ours。这些模型在12秒、24秒、36秒、48秒和60秒的生成效果可视化，展示了MemFlow在长视频叙事中的优势。](images/3.jpg)
    *该图像是示意图，展示了不同视频生成模型在不同时间点的输出效果，包括SkyReels-V2、FramePack、Self-Forcing、LongLive和Ours。这些模型在12秒、24秒、36秒、48秒和60秒的生成效果可视化，展示了MemFlow在长视频叙事中的优势。*

### 6.1.2. Single-prompt Generation
The paper also evaluates `MemFlow` on the more standard task of generating a video from a single, unchanging prompt.

**Short Video (5 seconds):**
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">#Params</th>
<th rowspan="2">Resolution</th>
<th rowspan="2">Throughput (FPS) ↑</th>
<th colspan="3">Evaluation scores ↑</th>
</tr>
<tr>
<th>Total</th>
<th>Quality</th>
<th>Semantic</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7"><b>Diffusion models</b></td>
</tr>
<tr>
<td>LTX-Video [9]</td>
<td>1.9B</td>
<td>768×512</td>
<td>8.98</td>
<td>80.00</td>
<td>82.30</td>
<td>70.79</td>
</tr>
<tr>
<td>Wan2.1 [29]</td>
<td>1.3B</td>
<td>832×480</td>
<td>0.78</td>
<td>84.26</td>
<td>85.30</td>
<td>80.09</td>
</tr>
<tr>
<td colspan="7"><b>Autoregressive models</b></td>
</tr>
<tr>
<td>SkyReels-V2 [3]</td>
<td>1.3B</td>
<td>960×540</td>
<td>0.49</td>
<td>82.67</td>
<td>84.70</td>
<td>74.53</td>
</tr>
<tr>
<td>MAGI-1 [28]</td>
<td>4.5B</td>
<td>832×480</td>
<td>0.19</td>
<td>79.18</td>
<td>82.04</td>
<td>67.74</td>
</tr>
<tr>
<td>CausVid [39]</td>
<td>1.3B</td>
<td>832×480</td>
<td>17.0</td>
<td>81.20</td>
<td>84.05</td>
<td>69.80</td>
</tr>
<tr>
<td>NOVA [7]</td>
<td>0.6B</td>
<td>768×480</td>
<td>0.88</td>
<td>80.12</td>
<td>80.39</td>
<td>79.05</td>
</tr>
<tr>
<td>Pyramid Flow [19]</td>
<td>2B</td>
<td>640×384</td>
<td>6.7</td>
<td>81.72</td>
<td>84.74</td>
<td>69.62</td>
</tr>
<tr>
<td>Self Forcing, chunk-wise [15]</td>
<td>1.3B</td>
<td>832×480</td>
<td>17.0</td>
<td>84.31</td>
<td>85.07</td>
<td>81.28</td>
</tr>
<tr>
<td>Self Forcing, frame-wise [15]</td>
<td>1.3B</td>
<td>832×480</td>
<td>8.9</td>
<td>84.26</td>
<td>85.25</td>
<td>80.30</td>
</tr>
<tr>
<td>LongLive [35]</td>
<td>1.3B</td>
<td>832×480</td>
<td>20.3†</td>
<td>84.87</td>
<td>86.97</td>
<td>76.47</td>
</tr>
<tr>
<td><b>MEMFLOW</b></td>
<td>1.3B</td>
<td>832×480</td>
<td><b>18.7</b></td>
<td><b>85.14</b></td>
<td>85.95</td>
<td><b>81.90</b></td>
</tr>
</tbody>
</table>

*   `MemFlow` achieves the highest `Total Score` (85.14) and `Semantic Score` (81.90), demonstrating that its prompt-aware memory helps improve text alignment even for short videos.
*   Its speed (18.7 FPS) is very competitive, only slightly slower than its memory-free baseline `LongLive` (20.3 FPS), confirming the high efficiency of the `SMA` mechanism.

**Long Video (30 seconds):**
The following are the results from Table 4 of the original paper:

| Model             | Total Score ↑ | Quality Score ↑ | Semantic Score ↑ | Throughput (FPS) ↑ |
|-------------------|---------------|-----------------|------------------|--------------------|
| SkyReels-V2 [3]   | 75.29         | 80.77           | 53.37            | 0.49               |
| FramePack [42]    | 81.95         | 83.61           | 75.32            | 0.92               |
| Self Forcing [15] | 81.59         | 83.82           | 72.70            | 17.0               |
| LongLive [35]     | 83.52         | 85.44           | 75.82            | 20.3               |
| **MEMFLOW**       | **84.51**     | **85.92**       | **78.87**        | **18.7**           |

*   Here, `MemFlow`'s advantages become even more pronounced. It outperforms all baselines across all three scores (`Total`, `Quality`, `Semantic`), showing that its memory mechanism is increasingly beneficial as the video duration extends.

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Ablation on Memory Mechanism
This study (Table 3) directly compares the key memory designs.

The following are the results from the ablation table (labeled Table 3 in the paper) of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Memory Mechanism</th>
<th rowspan="2">Subject Consistency ↑</th>
<th rowspan="2">Background Consistency ↑</th>
<th rowspan="2">Throughput (FPS) ↑</th>
<th colspan="6">CLIP Score ↑</th>
</tr>
<tr>
<th>0-10 s</th>
<th>10-20 s</th>
<th>20-30 s</th>
<th>30-40 s</th>
<th>40-50 s</th>
<th>50-60 s</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o Memory</td>
<td>94.41</td>
<td>95.15</td>
<td>23.5</td>
<td>26.74</td>
<td>25.10</td>
<td>24.60</td>
<td>23.61</td>
<td>24.23</td>
<td>24.14</td>
</tr>
<tr>
<td>Frame Sink [35]</td>
<td>97.66</td>
<td>96.20</td>
<td>20.3</td>
<td>26.63</td>
<td>25.77</td>
<td>24.65</td>
<td>23.99</td>
<td>24.52</td>
<td>24.11</td>
</tr>
<tr>
<td><b>NAM+SMA</b></td>
<td><b>98.01</b></td>
<td><b>96.70</b></td>
<td>18.7</td>
<td>26.31</td>
<td>24.70</td>
<td>23.94</td>
<td>24.13</td>
<td>24.90</td>
<td>24.22</td>
</tr>
<tr>
<td>NAM</td>
<td>98.05</td>
<td>96.57</td>
<td>17.6</td>
<td>26.50</td>
<td>25.30</td>
<td>24.42</td>
<td>24.23</td>
<td>24.96</td>
<td>24.28</td>
</tr>
</tbody>
</table>

*   **`w/o Memory` vs. Others:** Adding any form of memory (`Frame Sink` or `NAM`) significantly improves both `Subject` and `Background Consistency` compared to the memory-free baseline. This confirms that memory is essential.
*   **`Frame Sink` vs. `NAM`:** `NAM` (our full model, $NAM+SMA$) achieves the highest consistency scores (98.01 Subject, 96.70 Background), outperforming the simpler `Frame Sink` strategy of `LongLive`. This validates the superiority of the dynamic, semantic retrieval approach.
*   **`NAM` vs. $NAM+SMA$:** The `NAM` only version (using the full memory bank without sparse activation) achieves marginally better consistency and CLIP scores but at a lower speed (17.6 FPS). The full $NAM+SMA$ model recovers most of the performance while improving speed to 18.7 FPS. This shows `SMA` effectively balances the trade-off between quality and efficiency.

    The qualitative comparison in Figure 4 further illustrates these findings. The "w/o Memory" model has abrupt transitions, "Frame Sink" fails to maintain consistency for new subjects, while `MemFlow` ($NAM+SMA$) handles the narrative smoothly.

    ![该图像是示意图，展示了不同时间点（12s、24s、36s、48s、60s）在视频生成中使用不同内存设计（无内存、帧汇聚、NAM和NAM+SMA）的效果。每列代表不同的生成方法，显示出内存机制对视频内容一致性的影响。](images/4.jpg)
    *该图像是示意图，展示了不同时间点（12s、24s、36s、48s、60s）在视频生成中使用不同内存设计（无内存、帧汇聚、NAM和NAM+SMA）的效果。每列代表不同的生成方法，显示出内存机制对视频内容一致性的影响。*

### 6.2.2. Ablation on Memory Capacity
This study (Figure 5) investigates how the size of the memory bank affects performance.

![Figure 5. Quantitative analysis of different memory capacity under multi-prompt 60-second setting. "w/o Memory" means only attending to the local attention window, "Frame Sink" refers to keeping KV cache from the first chunk as memory \[35\], "NAM" adopts the whole memory bank including $^ { b }$ latent frames.](images/5.jpg)
*该图像是一个图表，展示了在多提示60秒设置下不同记忆容量的定量分析。图中对比了"None"、"Frame Sink"与不同参数$b$值的NAM（$b=3, 6, 9$）在不同时间段（0-10s, 10-20s, 20-30s等）的CLIP得分变化。*

*   The authors test `NAM` with memory bank sizes $b = \{3, 6, 9\}$ frames.
*   The results show that a larger memory is not always better. `NAM` with $b=6$ and $b=9$ show unstable performance and sometimes underperform the simpler `Frame Sink` baseline.
*   The authors hypothesize this is due to an **imbalance** in the model's attention. With a very large memory, the model might over-rely on global context from memory and neglect the crucial short-term context from the local window, disrupting the narrative flow.
*   The best and most stable performance is achieved with $b=3$, which is half the size of the local context window ($n=6$, though not explicitly stated, this is a common setting). This suggests that a balanced attention field between local and global context is optimal.

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper introduces `MemFlow`, a novel and effective framework for generating consistent and efficient long videos, especially in interactive, multi-prompt settings. Its main contributions are two synergistic mechanisms:
1.  **Narrative Adaptive Memory (NAM):** A dynamic memory update strategy that uses text prompts to retrieve semantically relevant historical frames, ensuring the model's memory is always aligned with the current narrative needs.
2.  **Sparse Memory Activation (SMA):** An efficient attention mechanism that activates only the most relevant parts of the memory on-the-fly, drastically reducing computational overhead without significant quality loss.

    Experiments demonstrate that `MemFlow` achieves state-of-the-art performance, outperforming previous methods in long-term consistency, visual quality, and narrative coherence, all while maintaining high generation speed (18.7 FPS).

## 7.2. Limitations & Future Work
The paper itself does not explicitly state its limitations. However, based on the methodology and results, we can infer some potential areas for future work:
*   **Richer Memory Representation:** The memory prototype for a new chunk is simply the first frame's `KV cache`. While efficient, this might be overly simplistic. Future work could explore more sophisticated summarization techniques to create a richer, more comprehensive prototype for each chunk.
*   **Scalability of Memory Bank:** The ablation study showed that simply increasing the memory bank size can be detrimental. Future research could investigate more advanced methods for managing a larger memory bank, perhaps with hierarchical structures or different attention patterns, to leverage more historical context without disrupting the local-global balance.
*   **Beyond Text Prompts:** The retrieval is currently guided by text prompts. This is powerful but might not capture all desired consistencies (e.g., subtle stylistic elements not described in text). Future work could incorporate other signals, such as visual similarity or motion cues, into the retrieval process.
*   **Generalization:** The model was trained on a specific type of narrative dataset. Its performance on highly complex, non-narrative long videos (e.g., documentaries, abstract visuals) remains an open question.

## 7.3. Personal Insights & Critique
*   **Elegance in Simplicity:** The core ideas behind `MemFlow` are both powerful and elegantly simple. Using the text prompt—an already available signal—to guide memory retrieval is a very intuitive and clever solution to the problem of adaptive context. Similarly, the `Redundant Removal` heuristic (using the first frame) and the mean-pooling descriptors in `SMA` are highly efficient choices that prove to be effective.
*   **Practicality and Impact:** The paper's focus on efficiency is a significant strength. By achieving high quality with a minimal speed penalty, `MemFlow` is not just an academic curiosity but a practical tool that could be deployed in real-world applications for interactive content creation. Its compatibility with existing `KV cache`-based models makes it easy to adopt.
*   **Critique on Memory Capacity:** The finding that a larger memory can hurt performance is fascinating and a crucial insight for the field. It highlights that long-range dependency is not just about having more context, but about having the *right* context and balancing it with short-term information. However, the paper's explanation is a hypothesis. A more in-depth analysis of the attention maps could provide concrete evidence for this "imbalance" and would have further strengthened the paper.
*   **Inspiration for Other Domains:** The core principle of `MemFlow`—using an external, guiding signal (the prompt) to dynamically shape the model's internal memory—is highly transferable. This concept could be applied to other long-sequence generation tasks, such as long-form story generation (where chapter summaries could guide memory), music generation (where a chord progression could guide note selection), or even in reinforcement learning for agents with long-term memory. The dual system of semantic retrieval and sparse activation is a general and powerful paradigm for managing large contextual histories.