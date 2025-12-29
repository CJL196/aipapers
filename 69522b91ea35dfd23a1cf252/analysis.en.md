# 1. Bibliographic Information

## 1.1. Title
**StoryMem: Multi-shot Long Video Storytelling with Memory**

## 1.2. Authors
The paper is authored by Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. The authors are primarily affiliated with the **S-Lab at Nanyang Technological University (NTU)** and **Intelligent Creation at ByteDance**. Liming Jiang served as the Project Lead, and Xingang Pan is the Corresponding Author.

## 1.3. Journal/Conference
This paper was published on **arXiv** (a prominent preprint server for rapid dissemination of research in Artificial Intelligence and Computer Vision) on **December 23, 2025**. Given the affiliation and the quality of the technical contribution, it is likely intended for a top-tier computer vision or machine learning conference (such as CVPR, ICCV, or NeurIPS).

## 1.4. Publication Year
**2025**

## 1.5. Abstract
Visual storytelling requires generating multi-shot videos with cinematic quality and long-range consistency. The authors propose **StoryMem**, a paradigm that reformulates long-form video storytelling as **iterative shot synthesis conditioned on explicit visual memory**. This transforms single-shot video diffusion models into multi-shot storytellers via a novel **Memory-to-Video (M2V)** design. This mechanism maintains a dynamically updated memory bank of keyframes from historical shots. These are injected into the model using **latent concatenation** and **negative RoPE shifts** with only **LoRA fine-tuning**. The system also includes a semantic keyframe selection strategy and aesthetic preference filtering. To evaluate this, they introduce **ST-Bench**, a benchmark for multi-shot storytelling. Results show superior cross-shot consistency and aesthetic quality compared to existing methods.

## 1.6. Original Source Link
*   **PDF Link:** [https://arxiv.org/pdf/2512.19539v1.pdf](https://arxiv.org/pdf/2512.19539v1.pdf)
*   **Project Page:** [https://kevin-thu.github.io/StoryMem](https://kevin-thu.github.io/StoryMem)

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The current state of video generation has reached "near-cinematic" fidelity for single, short clips (usually 2–10 seconds). However, the real goal of creators is **visual storytelling**, which requires multiple shots (clips) that maintain a consistent narrative, character identity, and environment over a minute or more.

The core problem is the **consistency-efficiency dilemma**:
1.  **Joint Modeling:** Training a massive model to generate all shots at once is computationally expensive (quadratic cost) and suffers from a lack of high-quality, long-form training data.
2.  **Decoupled Pipelines:** Generating shots independently based on keyframes leads to "rigid transitions" and "visual drift," where characters or scenes change noticeably between clips.

    The paper aims to find a "third path": adapting high-quality, single-shot models to be aware of the "history" of the story through an **explicit memory mechanism**, similar to how humans remember past events to understand a current narrative.

## 2.2. Main Contributions / Findings
*   **StoryMem Paradigm:** A framework that treats long video generation as an iterative process where each new shot is "conditioned" on a memory bank of past shots.
*   **Memory-to-Video (M2V) Design:** A technical method to inject memory frames into a standard video model using **negative RoPE shifts** (positional encodings that place memory in the "past").
*   **Dynamic Memory Management:** A strategy to select only the most "informative" and "aesthetically pleasing" frames for memory, preventing the model from becoming overwhelmed by redundant data.
*   **ST-Bench:** A new, rigorous benchmark specifically designed for multi-shot storytelling, featuring 30 diverse story scripts and 300 total prompts.
*   **Performance:** StoryMem significantly outperforms existing state-of-the-art models (like HoloCine) in **cross-shot consistency** (keeping characters/scenes the same) while maintaining higher visual quality.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, a novice needs to be familiar with several core AI technologies:

*   **Diffusion Models:** These are AI models that learn to generate data (like images or videos) by starting with random noise and gradually "denoising" it until it becomes a clear picture.
*   **Latent Video Diffusion (LVD):** Instead of working on every single pixel (which is computationally heavy), these models work on a "compressed" version of the video called a **latent space**. A **VAE (Variational AutoEncoder)** is used to compress and decompress the video.
*   **Transformers & DiT (Diffusion Transformer):** A `Transformer` is an architecture that uses `Attention` to weigh the importance of different parts of the input. A `DiT` is a specific type of Transformer used as the backbone for diffusion models.
*   **RoPE (Rotary Position Embedding):** Transformers don't inherently know the order of tokens (or frames). `RoPE` is a mathematical way to encode the position (time and space) of each pixel/frame into the model so it knows what comes before what.
*   **LoRA (Low-Rank Adaptation):** A technique for fine-tuning large models efficiently. Instead of updating all billions of parameters, you only update a small "sandwich" of new parameters, saving time and memory.

## 3.2. Previous Works
The paper identifies two main categories of prior research:
1.  **Keyframe-based Story Generation:** Models like `StoryDiffusion` or `StoryAgent` generate a storyboard of images first and then turn each image into a video clip. The calculation for the `Attention` mechanism in these models often uses **Consistent Self-Attention**, which can be simplified as:
    $ \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right)V $
    where $Q$ (Query), $K$ (Key), and $V$ (Value) are derived from multiple images simultaneously to ensure they look similar. However, these lack temporal awareness between clips.
2.  **Multi-shot Long Video Generation:** Models like `LCT` and `HoloCine` try to generate everything at once. `LCT` uses **3D RoPE** to model dependencies across a long sequence. While powerful, the "quadratic cost" means that doubling the video length quadruples the memory required.

## 3.3. Technological Evolution
The field started with **Image Generation** (Stable Diffusion), moved to **Single-Shot Video** (Sora, Kling, Wan2.2), and is now entering the **Long-Form Narrative** phase. StoryMem positions itself as an evolution of single-shot models, adding a "memory" layer rather than building a new giant model from scratch.

## 3.4. Differentiation Analysis
Unlike `HoloCine` (which requires massive multi-shot training data), StoryMem can be trained on **single-shot clips** by simply pretending some frames are "memory." Unlike `StoryDiffusion`, it maintains a "dynamic" memory that updates as the story progresses, allowing for shifting camera angles and evolving scenes.

---

# 4. Methodology

## 4.1. Principles
The core intuition is that a video model should "look back" at previous key moments to decide how to draw the current scene. StoryMem stores these moments in a **Memory Bank** and uses them as a global conditioning signal.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Problem Formulation
The authors formalize the story generation task as an **autoregressive** process. Instead of generating the whole video $\mathcal{V}$ at once, they generate it shot by shot.
The joint distribution of the video $\mathcal{V}$ given a script $\mathcal{T}$ is:
$ p_{\Theta}(\mathcal{V} | \mathcal{T}) \approx \prod_{i=1}^{N} p_{\Theta}(v_i | t_i, m_{i-1}) $

*   $v_i$: The current video shot.
*   $t_i$: The text description for that shot.
*   $m_{i-1}$: The **memory** of all shots generated up to that point.

    The goal is to update the memory $m_i$ after each shot:
$ m_i = f_{\phi}(m_{i-1}, v_i) $
where $f_{\phi}$ is the memory update function.

The following figure (Figure 2 from the original paper) shows the system architecture:

![Figure 2 Overview of StoryMem. StoryMem generates each shot conditioned on a memory bank that stores keams fom previusy nrate shots. Duri eneration, the seecmemory ame are ncoded b 3D VAE, fused with noisy video latents and binary masks, and fed into a LoRA-finetuned memory-conditioned Video DiT to apivo narrative progression. Byerativey enerating shots wihmemory updatesStoryMem produce coherent minue-n, multi-shot story videos.](images/2.jpg)
*该图像是示意图，展示了StoryMem的工作流程。它描述了如何通过内存银行生成视频镜头，其中包括3D VAE编码器、LoRA微调以及视觉记忆的更新过程。此外，还涉及到语义关键帧选择和美学偏好过滤，以提升生成的视频质量。*

### 4.2.2. Memory-to-Video (M2V) with Negative RoPE Shift
To make the model aware of the memory, the authors use two main tricks:

1.  **Latent Concatenation:** They take the memory frames, encode them into the latent space using a **3D VAE**, and "glue" them to the start of the current shot's noisy latents.
2.  **Negative RoPE Shift:** This is the most innovative part. Standard positional encodings start at index 0. If you just add memory frames, the model might think they are part of the current clip. To fix this, they assign **negative indices** to the memory.
    The temporal indices are defined as:
    $ \{ - f _ { m } S , - ( f _ { m } - 1 ) S , \ldots , - S , 0 , 1 , \ldots , f - 1 \} $
    *   $f_m$: Number of memory frames.
    *   $f$: Number of current shot frames.
    *   $S$: A fixed **offset** (gap) that tells the model these memory frames happened in the past.

        This allows the Transformer's `Attention` mechanism to "look back" at negative time indices to find consistent visual cues while keeping the current shot's timeline starting at 0.

### 4.2.3. Memory Extraction and Update
Not every frame is worth remembering. The authors use a two-step filter:
1.  **Semantic Keyframe Selection:** They use **CLIP** (a model that understands the relationship between images and text) to compute embeddings. A new frame is only added to memory if its "cosine similarity" to existing memory frames is low (meaning it shows something new).
2.  **Aesthetic Preference Filtering:** They use **HPSv3** (Human Preference Score) to ensure that only clear, high-quality frames are stored. If a frame is blurry or distorted, it is discarded to prevent "garbage in, garbage out" for future shots.

### 4.2.4. MI2V and MR2V Extensions
The framework is flexible:
*   **MI2V (Memory + Image-to-Video):** If two shots are in the same scene, the last frame of shot $A$ is used as the first frame of shot $B$. This ensures a perfectly smooth transition.
*   **MR2V (Memory + Reference-to-Video):** Users can upload a photo of themselves or a specific character, which is placed into the initial memory $m_0$ to ensure the AI generates *that* specific person throughout the story.

    ---

# 5. Experimental Setup

## 5.1. Datasets
*   **Training Data:** 400,000 single-shot videos (5 seconds each). They "simulated" stories by grouping visually similar clips together during training.
*   **ST-Bench (Evaluation):** A new benchmark created with **GPT-5**. It contains 30 scripts, each with 8–12 shots. Styles include realistic, fairy-tale, and cinematic.

## 5.2. Evaluation Metrics
The authors use four key metrics:

1.  **Aesthetic Quality:** Measures how "beautiful" or "cinematic" the video is.
    *   **Calculation:** Uses a predictor trained on the LAION dataset to score visual appeal.
2.  **Prompt Following (ViCLIP):** Measures how well the video matches the text script.
    *   **Formula:** $ \mathrm{sim}(v, t) = \frac{\Phi(v) \cdot \Psi(t)}{\|\Phi(v)\| \|\Psi(t)\|} $
    *   $\Phi(v)$ is the video embedding, $\Psi(t)$ is the text embedding.
3.  **Cross-shot Consistency:** Measures if the character/scene looks the same across the whole video.
    *   **Calculation:** The average ViCLIP similarity between all pairs of shots in a story.
4.  **Top-10 Consistency:** A more refined version of the above, focusing on the 10 most relevant shot pairs to see if specific character details are preserved.

## 5.3. Baselines
The method is compared against:
*   **Wan2.2 (Independent):** The base model generating shots one by one with no memory.
*   **StoryDiffusion:** A keyframe-based method.
*   **IC-LoRA:** An in-context image generation method.
*   **HoloCine:** The previous state-of-the-art for "joint" long video generation.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
StoryMem outperformed all baselines in **Cross-shot Consistency** by a significant margin (nearly 30% better than the base model). It also maintained the highest **Aesthetic Quality**, proving that adding memory doesn't "break" the visual beauty of the original model.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Aesthetic Quality ↑</th>
<th colspan="2">Prompt Following ↑</th>
<th colspan="2">Cross-shot Consistency ↑</th>
</tr>
<tr>
<th>Global</th>
<th>Single-shot</th>
<th>Overall</th>
<th>Top-10 Pairs</th>
</tr>
</thead>
<tbody>
<tr>
<td>Wan2.2 (Independent)</td>
<td>0.6097</td>
<td>0.2114</td>
<td>0.2335</td>
<td>0.3934</td>
<td>0.4357</td>
</tr>
<tr>
<td>StoryDiffusion + Wan2.2</td>
<td>0.6053</td>
<td>0.2223</td>
<td>0.2294</td>
<td>0.4485</td>
<td>0.4902</td>
</tr>
<tr>
<td>IC-LoRA + Wan2.2</td>
<td>0.6001</td>
<td>0.2198</td>
<td>0.2251</td>
<td>0.4571</td>
<td>0.5015</td>
</tr>
<tr>
<td>HoloCine</td>
<td>0.5471</td>
<td>0.2104</td>
<td>0.2155</td>
<td>0.4631</td>
<td>0.5122</td>
</tr>
<tr>
<td><b>StoryMem (Ours)</b></td>
<td><b>0.6133</b></td>
<td><b>0.2289</b></td>
<td>0.2313</td>
<td><b>0.5065</b></td>
<td><b>0.5337</b></td>
</tr>
</tbody>
</table>

### 6.1.1. Qualitative Analysis
As seen in Figure 3, StoryMem keeps the character's outfit and the street environment consistent even in shot 5, whereas other models have already "forgotten" what the street looked like in shot 2.

![Figure 3 Qualitative comparison. Our StoryMem generates coherent multi-scene, multi-shot story videos aligned w per-ho eptions.n ontrast,he reraimode ankeambasbaselnes preeveo character and scene consistency, while HoloCine \[31\] exhibits noticeable degradation in visual quality.](images/3.jpg)
*该图像是一个示意图，展示了使用StoryMem生成的多场景多镜头故事视频。这些镜头展现了不同的场景，分别采用了各种拍摄技巧和镜头长度，以突显视频的连贯性和美学品质。图中比较了不同方法的输出效果，包括HoloCine的表现。*

## 6.2. Ablation Studies
The authors tested what happens if they remove parts of their system.
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Aesthetic Quality ↑</th>
<th colspan="2">Prompt Following ↑</th>
<th colspan="2">Cross-shot Consistency ↑</th>
</tr>
<tr>
<th>Global</th>
<th>Single-shot</th>
<th>Overall</th>
<th>Top-10 Pairs</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o Semantic Selection</td>
<td>0.6076</td>
<td>0.2257</td>
<td>0.2295</td>
<td>0.4878</td>
<td>0.5287</td>
</tr>
<tr>
<td>w/o Aesthetic Filtering</td>
<td>0.6018</td>
<td>0.2251</td>
<td>0.2313</td>
<td>0.4844</td>
<td>0.5330</td>
</tr>
<tr>
<td>w/o Memory Sink</td>
<td>0.6093</td>
<td>0.2277</td>
<td>0.2330</td>
<td>0.4891</td>
<td>0.5241</td>
</tr>
<tr>
<td><b>Ours (Full)</b></td>
<td><b>0.6133</b></td>
<td><b>0.2289</b></td>
<td>0.2313</td>
<td><b>0.5065</b></td>
<td><b>0.5337</b></td>
</tr>
</tbody>
</table>

*   **Semantic Selection:** Essential for adding new characters. Without it, if a character appears later in a clip, the model might not "record" them into memory.
*   **Aesthetic Filtering:** Prevents the model from trying to stay consistent with a "bad" or "glitchy" frame.

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
StoryMem represents a significant leap in **long-form video generation**. By moving away from "all-at-once" generation and toward a "memory-conditioned iterative" approach, the authors solved the problem of consistency without the massive computational overhead of other models. Their use of **Negative RoPE Shifts** is a clever architectural trick that allows standard video models to perceive the past without confusing it with the present.

## 7.2. Limitations & Future Work
*   **Multi-character Confusion:** If a scene has five different people, the "pure visual memory" can get confused about who is who. The authors suggest that adding **textual meta-information** (e.g., "this is the memory of the Professor") to the memory bank would help.
*   **Motion Discrepancy:** If shot A is very fast and shot B is very slow, the transition can still look a bit jumpy. Future work will look at "overlapping" more frames to smooth out these speed differences.

## 7.3. Personal Insights & Critique
The "Negative RoPE Shift" is the "Aha!" moment of this paper. It's a elegant mathematical solution to a temporal problem. Most researchers try to solve consistency by making the Attention window larger, which is expensive. StoryMem simply changes the "address" of the data so the model knows it’s looking at history. 

One potential issue is the "Memory Sink." If a story is 10 minutes long, even a compressed memory bank might become too large or lose focus. The current "sliding window" approach might eventually lead to the model "forgetting" the beginning of a very long movie. However, for the current goal of ~1 minute videos, this is the most practical and high-quality solution available.