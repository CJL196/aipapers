# 1. Bibliographic Information

## 1.1. Title
Memorize-and-Generate: Towards Long-Term Consistency in Real-Time Video Generation

## 1.2. Authors
Tianrui Zhu, Shiyi Zhang, Zhirui Sun, Jingqi Tian, and Yansong Tang. The authors are affiliated with the Tsinghua Shenzhen International Graduate School, Tsinghua University. Tianrui Zhu and Shiyi Zhang contributed equally as lead researchers.

## 1.3. Journal/Conference
This paper was published as a preprint on arXiv on December 21, 2024 (with a second version on December 25, 2024). While currently a preprint, the work addresses high-impact topics in computer vision and real-time generative AI, fields typically associated with top-tier conferences like CVPR or NeurIPS.

## 1.4. Publication Year
2025 (Published UTC: 2025-12-21)

## 1.5. Abstract
The paper addresses the challenge of maintaining long-term consistency in real-time video generation. While `frame-level autoregressive` (frame-AR) models have enabled real-time performance, they often suffer from "catastrophic forgetting"—losing track of previous scenes when the camera pans away—because they rely on short `attention windows` to save memory. The authors propose **Memorize-and-Generate (MAG)**, a framework that decouples memory compression from frame generation. It uses a dedicated **memory model** to compress history into a compact `KV cache` and a **generator model** to synthesize new frames. They also introduce **MAG-Bench** to evaluate historical scene retention. Results show that MAG achieves superior consistency and high-speed generation (16-21 FPS).

## 1.6. Original Source Link
*   **Official ArXiv Link:** [https://arxiv.org/abs/2512.18741](https://arxiv.org/abs/2512.18741)
*   **PDF Link:** [https://arxiv.org/pdf/2512.18741v2.pdf](https://arxiv.org/pdf/2512.18741v2.pdf)

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The field of video generation is shifting from slow, `bidirectional diffusion models` (which look at the whole video at once) to `autoregressive` models (which generate frame-by-frame). While `autoregressive` models are faster, they face a massive trade-off:
1.  **Memory Bloat:** Keeping every past frame in memory (the `KV cache`) quickly exhausts GPU resources.
2.  **Forgetting:** To save memory, current models use a "sliding window," only looking at the last few seconds. This causes the model to "forget" what a scene looked like once the camera moves away, leading to inconsistency when the camera returns.

    The core problem is **historical scene consistency**. If a camera pans left and then returns right, the original scene should still be there. Most current models fail this test.

## 2.2. Main Contributions / Findings
*   **MAG Framework:** A two-stage paradigm that separates "remembering" from "generating." A `memory model` compresses past data, and a `generator` uses that compressed data to produce new frames.
*   **Enhanced Training Objective:** They identify a "degenerate solution" where models ignore history and just follow text prompts. They fix this by training the model to generate frames based *only* on history (using empty text prompts).
*   **Memory Compression:** They achieve a $3\times$ compression of the `KV cache` with near-lossless reconstruction, significantly reducing GPU memory requirements.
*   **MAG-Bench:** A new benchmark specifically designed to test whether a model remembers a scene after the camera has moved away and returned.
*   **Efficiency:** The model generates video at roughly 21.7 FPS on an H100 GPU, making it suitable for real-time applications like game engines.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several core concepts in modern AI:

*   **Autoregressive (AR) Generation:** A process where the model predicts the next element in a sequence (like a video frame) based on all previous elements.
*   **KV Cache (Key-Value Cache):** In `Transformer` models, the "Key" and "Value" represent the memory of what the model has already processed. Storing this allows the model to avoid re-calculating everything for every new frame, but it takes up a lot of memory.
*   **Diffusion Models:** A class of generative models that create data by gradually removing noise from a random starting point.
*   **Flow Matching:** A recent alternative to standard diffusion that learns a straight-line path from noise to data, often requiring fewer steps to generate high-quality results.
*   **DMD (Distribution Matching Distillation):** A technique used to "distill" a slow, high-quality model (the teacher) into a fast, one-step or few-step model (the student).

## 3.2. Previous Works
*   **Wan2.1:** A state-of-the-art `bidirectional attention` model. It creates great short clips but is too slow for real-time streaming because it calculates relationships between every frame simultaneously.
*   **Self Forcing:** A training method that ensures the model's environment during training matches its environment during inference (generation). It uses `DMD` to make `autoregressive` models fast.
*   **LongLive / Self Forcing++:** Recent attempts to extend `Self Forcing` to longer videos. However, they mostly use `sliding windows`, which leads to the "forgetting" problem mentioned earlier.

## 3.3. Technological Evolution
Video generation has evolved from **U-Net** based diffusion to **Diffusion Transformers (DiT)**. While DiTs scale well, their memory usage grows quadratically with sequence length. This led to the development of `autoregressive` DiTs. This paper represents the next step: adding a "long-term memory" layer to these fast models without breaking the real-time speed.

## 3.4. Differentiation Analysis
Unlike previous works that simply discard old frames (`sliding window`) or try to update model weights during generation (`Test-Time Training`), **MAG** keeps all history but **compresses** it. It treats memory as a specific "reconstruction" task, ensuring that the compressed representation is actually high-quality enough to recreate the original scene.

---

# 4. Methodology

## 4.1. Principles
The core idea is **decoupling**. Instead of forcing one model to do everything, MAG uses:
1.  A **Memory Model** that focuses on "how to store this frame efficiently so I can recreate it later?"
2.  A **Generator Model** that focuses on "given this compressed memory and the current frame, what happens next?"

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Rethinking DMD Optimization
The authors start by analyzing the standard `Distribution Matching Distillation (DMD)` process. In standard `DMD`, a student model $S$ learns from a teacher $T$ by minimizing the difference between their output distributions. The gradient for this optimization is:

\$
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{DMD}} = \mathbb{E}_{\boldsymbol{x}} \left[ \nabla_{\boldsymbol{\theta}} \mathrm{KL} \left( p_{\boldsymbol{\theta}}^{\mathcal{S}}(\boldsymbol{x}) \parallel p^{\mathcal{T}}(\boldsymbol{x}) \right) \right] \approx \mathbb{E}_{i \sim U\{1, k\}} \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ s^{\mathcal{T}}(\boldsymbol{x}_i) - s_{\boldsymbol{\theta}}^{\mathcal{S}}(\boldsymbol{x}_i) \frac{d G_{\boldsymbol{\theta}}(\boldsymbol{z}_i)}{d \boldsymbol{\theta}} \right]
\$

Where:
*   $\boldsymbol{\theta}$: The parameters of the student model.
*   $p_{\boldsymbol{\theta}}^{\mathcal{S}}$ and $p^{\mathcal{T}}$: The probability distributions of the student and teacher, respectively.
*   $\boldsymbol{x}_i$: A video clip sampled from the dataset.
*   $\boldsymbol{z}_i$: Gaussian noise used as the starting point for generation.
*   $s^{\mathcal{T}}$ and $s_{\boldsymbol{\theta}}^{\mathcal{S}}$: The "score functions" (gradients of the log-probability) of the teacher and student.

    The authors identify a problem: if the teacher $T$ only knows how to generate video from text ($T$), and not from history ($h$), the student might learn a "shortcut." It might just ignore the history $h$ and generate frames based solely on the text $T$. To fix this, they introduce a **history loss** $\mathcal{L}_{\mathrm{history}}$ using an empty text condition $\boldsymbol{\vartheta}$:

\$
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{history}} = \mathbb{E}_{\boldsymbol{x} \sim p_{\boldsymbol{\theta}}^G(\boldsymbol{x} | h, \boldsymbol{\vartheta})} [\nabla_{\boldsymbol{\theta}} D_{KL} (p_{\boldsymbol{\theta}}^S(\boldsymbol{x}) \| p^{\mathcal{T}}(\boldsymbol{x}))]
\$

The final training objective becomes a balance:

\$
\nabla_{\boldsymbol{\theta}} \mathcal{L} = (1 - \lambda) \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{DMD}} + \lambda \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{history}}
\$

Where $\lambda$ is a hyperparameter that weights the importance of the history-based generation. By using an empty text prompt ($\boldsymbol{\vartheta}$), the model is **forced** to look at the history $h$ to figure out what to draw next.

### 4.2.2. The Memory Model (Stage 1)
The Memory Model is trained to act like an `Autoencoder` for the `KV cache`. In a streaming video, frames are processed in small "blocks" (e.g., 3 frames). 

The following figure (Figure 2 from the paper) illustrates this two-stage pipeline:

![Fig. 2: The training pipeline. The training process of MAG comprises two stages. In the first stage, we train the memory model for the triple compressed KV cache, retaining only one frame within a full attention block. The loss function requires the model to reconstruct the pixels of all frames in the block from the compressed cache. The process utilizes a customized attention mask to achieve efficient parallel training. In the second stage, we train the generator model within the long video DMD training framework to adapt to the compressed cache provided by the frozen memory model.](images/2.jpg)

**The Process:**
1.  **Encoding:** The model uses `full attention` within a block of frames to compress the information of all frames into the `KV cache` of just **one** frame (typically the last one).
2.  **Decoding:** The model is then tasked with reconstructing the actual pixels of **all** frames in that block using *only* that compressed `KV cache` and some random noise.
3.  **Parallel Training:** To make this efficient, they use a custom `attention mask` shown in Figure 3.

    The following figure (Figure 3) explains the attention masking strategy:

    ![Fig. 3: The attention mask of memory model training. We achieve efficient parallel training of the encode-decode process by concatenating noise and clean frame sequences. By masking out the KV cache of other frames within the block, the model is forced to compress information into the target cache.](images/3.jpg)
    *该图像是示意图，展示了记忆模型训练的注意力掩码。通过将噪声和干净帧序列连接，实现了编码-解码过程的高效并行训练。图中通过屏蔽出块内其他帧的KV缓存，迫使模型将信息压缩到目标缓存中。*

By masking out the cache of other frames, the model has no choice but to "squeeze" all the important visual information into the target cache.

### 4.2.3. Streaming Generator Training (Stage 2)
Once the Memory Model is frozen, the Generator Model is trained. It learns to take the compressed `KV cache` provided by the Memory Model and generate the next frame. Because the Memory Model can reconstruct the original frames nearly perfectly, the Generator has access to high-fidelity historical context without the massive memory cost of uncompressed frames.

---

# 5. Experimental Setup

## 5.1. Datasets
*   **VPData:** A dataset containing 390,000 high-quality real-world videos. This is used to train the `Memory Model` to ensure it can handle various textures and movements.
*   **VidProM:** A dataset of 1 million real prompts used to train the `Generator Model` for text-to-video alignment.
*   **MAG-Bench (The Authors' Dataset):** 176 videos with "backtracking" camera movements (panning away and coming back).

    The following figure (Figure 4) shows examples from the MAG-Bench dataset:

    ![Fig. 4: Examples from MAG-Bench. MAG-Bench is a lightweight benchmark comprising 176 videos featuring indoor, outdoor, object, and video game scenes. The benchmark also provides appropriate switch times to guide the model toward correct continuation using a few frames.](images/4.jpg)
    *该图像是示意图，展示了MAG-Bench中的视频生成过程示例，包括左侧的输入帧变化和右侧的测试结果，分别展示了不同的视角变化（如平移与缩放）。这些场景有助于评估模型在记忆和缓存管理上的表现。*

## 5.2. Evaluation Metrics
The authors use several metrics to evaluate quality and consistency:

1.  **PSNR (Peak Signal-to-Noise Ratio):**
    *   **Definition:** Measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher is better.
    *   **Formula:** `PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)`
    *   **Symbols:** $MAX_I$ is the maximum possible pixel value (e.g., 255); `MSE` is the Mean Squared Error.
2.  **SSIM (Structural Similarity Index Measure):**
    *   **Definition:** Quantifies the degradation of image quality caused by processing, focusing on luminance, contrast, and structure.
    *   **Formula:** $SSIM(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$
    *   **Symbols:** $\mu$ is the mean; $\sigma^2$ is the variance; $\sigma_{xy}$ is the covariance; $c$ are constants to stabilize division.
3.  **LPIPS (Learned Perceptual Image Patch Similarity):**
    *   **Definition:** Uses a pre-trained deep network to judge how "similar" two images look to a human. Lower is better.
    *   **Formula:** $d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l) \|_2^2$
    *   **Symbols:** $\hat{y}^l$ are activations from layer $l$ of a network; $w_l$ are weights.

## 5.3. Baselines
The model is compared against:
*   **Wan2.1:** The base non-autoregressive model.
*   **Self Forcing / Self Forcing++:** The standard for distilled autoregressive video.
*   **LongLive:** A model using a sliding window for real-time long video.
*   **SkyReels-V2:** A non-distilled autoregressive model.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
MAG achieves the best balance between speed, quality, and memory.
*   **Speed:** 21.7 FPS (Faster than all baselines).
*   **Memory:** $3\times$ compression allows for much longer context.
*   **Consistency:** On MAG-Bench, it significantly outperforms others, showing it doesn't "hallucinate" new objects when returning to a previous scene.

## 6.2. Data Presentation (Tables)

The following are the results from Table 1 of the original paper, comparing performance on 5-second videos:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Throughput FPS↑</th>
<th colspan="5">Vbench scores on 5s ↑</th>
</tr>
<tr>
<th>Total</th>
<th>Quality</th>
<th>Semantic</th>
<th>Background</th>
<th>Subject</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Multi-step model</strong></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>SkyReels-V2 [5]</td>
<td>0.49</td>
<td>82.67</td>
<td>84.70</td>
<td>74.53</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Wan2.1 [43]</td>
<td>0.78</td>
<td>84.26</td>
<td>85.30</td>
<td>80.09</td>
<td>97.29</td>
<td>96.34</td>
</tr>
<tr>
<td><strong>Few-step distillation model</strong></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>CausVid [52]</td>
<td>17.0</td>
<td>82.46</td>
<td>83.61</td>
<td>77.84</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Self Forcing [20]</td>
<td>17.0</td>
<td>83.98</td>
<td>84.75</td>
<td>80.86</td>
<td>96.21</td>
<td>96.80</td>
</tr>
<tr>
<td>Self Forcing++ [9]</td>
<td>17.0</td>
<td>83.11</td>
<td>83.79</td>
<td>80.37</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Longlive [48]</td>
<td>20.7</td>
<td>83.32</td>
<td>83.99</td>
<td>80.68</td>
<td>96.41</td>
<td>96.54</td>
</tr>
<tr>
<td><strong>MAG (Ours)</strong></td>
<td><strong>21.7</strong></td>
<td>83.52</td>
<td>84.11</td>
<td>81.14</td>
<td><strong>97.44</strong></td>
<td><strong>97.02</strong></td>
</tr>
</tbody>
</table>

The following are the results from Table 3, showing performance on the historical consistency benchmark (MAG-Bench):

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">History Context Comparison</th>
<th colspan="3">Ground Truth Comparison</th>
</tr>
<tr>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Self Forcing [20]</td>
<td>14.46</td>
<td>0.48</td>
<td>0.49</td>
<td>15.65</td>
<td>0.51</td>
<td>0.42</td>
</tr>
<tr>
<td>CausVid [52]</td>
<td>15.13</td>
<td>0.50</td>
<td>0.41</td>
<td>17.21</td>
<td>0.56</td>
<td>0.31</td>
</tr>
<tr>
<td>Longlive [48]</td>
<td>16.42</td>
<td>0.53</td>
<td>0.32</td>
<td>18.92</td>
<td>0.62</td>
<td>0.22</td>
</tr>
<tr>
<td>w/o stage 1</td>
<td>17.19</td>
<td>0.54</td>
<td>0.31</td>
<td>19.04</td>
<td>0.60</td>
<td>0.22</td>
</tr>
<tr>
<td><strong>MAG (Ours)</strong></td>
<td><strong>18.99</strong></td>
<td><strong>0.60</strong></td>
<td><strong>0.23</strong></td>
<td><strong>20.77</strong></td>
<td><strong>0.66</strong></td>
<td><strong>0.17</strong></td>
</tr>
</tbody>
</table>

## 6.3. Qualitative Visual Comparison
The superiority of the model is best seen in Figure 7, where competing models "forget" the geometry of the room as the camera pans back, while MAG maintains the structure.

![Fig. 7: Qualitative comparison on MAG-Bench. We primarily display the visual results of comparable distilled models. Prior to these frames, the models receive and memorize historical frames. Red boxes highlight instances of scene forgetting and hallucinations exhibited by other methods.](images/7.jpg)
*该图像是图表，展示了MAG模型与其他模型在MAG-Bench上的定性比较。上方为真实图像（GT），下方依次为MAG（我们的模型）、Self Forcing和Longlive。在红框中突出显示了其他方法所展示的场景遗忘和幻觉实例。*

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The MAG framework successfully addresses the conflict between **real-time performance** and **long-term historical consistency**. By decoupling memory into a compression task and a generation task, and by introducing a training loss that forces history reliance, the authors have created a model that can generate high-quality, minute-long videos with consistent backgrounds.

## 7.2. Limitations & Future Work
*   **Data Scarcity:** There is still a lack of massive datasets specifically designed for long-term consistency (like complex 3D camera paths).
*   **Teacher Reliance:** The distilled student model is still limited by the quality of the teacher model (Wan2.1).
*   **World Models:** The current framework isn't yet an "interactive world model" where a user can provide actions (like in a game), though the authors suggest this is a future direction.

## 7.3. Personal Insights & Critique
This paper provides a very elegant engineering solution to a fundamental Transformer problem: **KV Cache management**. Instead of trying to make the cache smaller through math-only tricks (like pruning), they treat the cache as a **visual latent space** that can be learned. 

One potential critique is the **two-stage training**. Training a dedicated memory model first adds complexity to the pipeline. However, the results in Table 3 (comparing MAG to "w/o stage 1") clearly show that simple downsampling doesn't work; the specialized memory training is necessary to keep the details that allow the model to "recognize" a scene it hasn't seen for 20 seconds. This is a significant step toward making generative AI useful for virtual environments and gaming.