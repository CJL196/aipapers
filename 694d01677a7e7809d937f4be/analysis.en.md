# 1. Bibliographic Information

## 1.1. Title
Memorize-and-Generate: Towards Long-Term Consistency in Real-Time Video Generation

The title clearly outlines the paper's core proposal and objective. "Memorize-and-Generate" (MAG) refers to the proposed two-stage framework. "Long-Term Consistency" points to the primary problem being addressed—maintaining scene integrity over extended video durations. "Real-Time Video Generation" specifies the application domain and performance constraint, indicating a focus on efficient, streaming-capable models.

## 1.2. Authors
Tianrui Zhu, Shiyi Zhang, Zhirui Sun, Jingqi Tian, and Yansong Tang.

All authors are affiliated with the Tsinghua Shenzhen International Graduate School at Tsinghua University. This indicates the research originates from a single academic institution known for its strong engineering and computer science programs.

## 1.3. Journal/Conference
The paper is available on arXiv, a preprint server for academic papers. The publication date is listed as December 21, 2025, which suggests this is a placeholder date. As a preprint, this work has not yet undergone formal peer review. arXiv is a common platform for researchers in fast-moving fields like machine learning to disseminate their work quickly.

## 1.4. Publication Year
2025 (as per the provided metadata).

## 1.5. Abstract
The abstract introduces frame-level autoregressive (`frame-AR`) models as a promising direction for real-time video generation, rivaling traditional bidirectional diffusion models. It identifies a key challenge in generating long videos: the trade-off between memory and consistency. Current methods either use `window attention`, which forgets past information and causes scene inconsistency, or attempt to retain the full history, which is computationally prohibitive.

To solve this, the authors propose **Memorize-and-Generate (MAG)**, a framework that decouples memory management from frame generation. MAG consists of two specialized models: a **memory model** trained to compress historical information into a compact Key-Value (`KV`) cache, and a **generator model** that uses this compressed memory to synthesize new frames. The paper also introduces **MAG-Bench**, a new benchmark designed to specifically evaluate a model's ability to retain historical information. Experiments show that MAG achieves superior long-term consistency while remaining competitive on standard video generation benchmarks.

## 1.6. Original Source Link
*   **Original Source Link:** `https://arxiv.org/abs/2512.18741`
*   **PDF Link:** `https://arxiv.org/pdf/2512.18741v2.pdf`
*   **Publication Status:** Preprint on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem this paper addresses is the challenge of **long-term consistency** in real-time, long-form video generation. As video generation models become more powerful, the goal has shifted from generating short, few-second clips to creating minute-level videos.

The dominant paradigm for this is **frame-level autoregressive (frame-AR) generation**, where video is created frame by frame or chunk by chunk. While this approach enables real-time, streaming generation, it faces a fundamental dilemma:

1.  **Limited Memory (Catastrophic Forgetting):** To manage computational costs, many models use a "sliding window" approach, only paying attention to the most recent few seconds of video history. This causes the model to "forget" anything that happened before the window. For example, if a camera pans away from an object and then returns, a model with a limited window might generate a completely different object or scene, breaking immersion and consistency.
2.  **Full Memory (Prohibitive Cost):** The alternative is to retain the entire history of generated frames. However, the memory required to store this information (specifically, the `KV cache` in Transformer-based models) grows linearly with the video length and quickly exceeds the capacity of modern GPUs. A one-minute video can require terabytes of memory, making this approach infeasible.

    This paper's entry point is an innovative solution to this trade-off. Instead of choosing between limited or full memory, the authors propose to **intelligently compress the full history**. Their key idea is to **decouple** the task into two specialized sub-problems: one of efficiently memorizing the past, and another of creatively generating the future based on that memory.

## 2.2. Main Contributions / Findings
The paper makes several key contributions:

1.  **The Memorize-and-Generate (MAG) Framework:** A novel two-stage architecture that separates the tasks of memory compression and frame generation. This decoupling allows each component to be optimized for its specific function, leading to a more efficient and effective overall system.
2.  **A Learnable Memory Compression Model:** They propose a dedicated `memory model` trained with an autoencoder-like objective to compress the information from a block of frames into a compact `KV cache` representation of a single frame. This achieves a significant compression ratio (e.g., 3x) with minimal loss of fidelity.
3.  **An Improved Training Objective for Consistency:** The authors identify a "degenerate solution" in existing training methods where models learn to rely on text prompts while ignoring historical context. They introduce a modified loss function that forces the `generator model` to generate content based solely on historical frames, thereby strengthening its ability to maintain consistency.
4.  **MAG-Bench: A New Benchmark for Historical Consistency:** To properly evaluate their method, they created `MAG-Bench`, a dataset of videos featuring "leave-and-return" camera movements. This provides a direct and quantitative way to measure a model's ability to remember scenes that have temporarily moved out of view.

    The main finding is that the **MAG framework successfully resolves the memory-consistency trade-off**. It achieves significantly better historical consistency than previous methods, as demonstrated on `MAG-Bench`, while maintaining competitive video quality and real-time inference speeds (21.7 FPS on an H100 GPU) on standard benchmarks.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that have become state-of-the-art in generating high-quality images and videos. The core idea is based on a two-step process:
*   **Forward Process (Noising):** Start with a clean data sample (e.g., an image) and gradually add a small amount of Gaussian noise over many steps, until the image becomes pure noise. This process is fixed and does not involve learning.
*   **Reverse Process (Denoising):** Train a neural network to reverse this process. The model takes a noisy image and a timestep as input and learns to predict the noise that was added at that step. By repeatedly applying this denoising model, one can start from pure random noise and generate a clean, realistic data sample.

### 3.1.2. Autoregressive (AR) Models
Autoregressive models generate data sequentially, where each new element is conditioned on the elements that came before it. For a sequence $x = (x_1, x_2, ..., x_n)$, the joint probability is factorized as:
\$
p(x) = p(x_1) \prod_{i=2}^{n} p(x_i | x_1, ..., x_{i-1})
\$
In **frame-AR video generation**, this means generating the current frame (or a small chunk of frames) based on all the previously generated frames. This is naturally suited for streaming applications, as one doesn't need to know the future to generate the present.

### 3.1.3. KV Cache in Transformers
Transformers are the dominant architecture for sequence modeling. Their core component is the `self-attention` mechanism. In self-attention, each token in a sequence creates three vectors: a **Query (Q)**, a **Key (K)**, and a **Value (V)**. To compute the output for a given token, its Query vector is compared with the Key vectors of all other tokens to calculate attention scores. These scores are then used to create a weighted sum of all Value vectors.

During autoregressive generation, when generating the $i$-th token, the model needs to attend to all previous tokens $1, ..., i-1$. Instead of re-computing the K and V vectors for all previous tokens at every step, we can store (cache) them. This stored set of Key and Value tensors is called the **KV cache**. At step $i$, we only need to compute the Q, K, and V for the new token and append its K and V to the cache. This drastically speeds up generation but is also the source of the memory problem, as the cache grows with the sequence length.

### 3.1.4. Distribution Matching Distillation (DMD) and Self Forcing
Generating high-quality video with diffusion models typically requires many denoising steps (e.g., 20-50), which is too slow for real-time applications. **Distribution Matching Distillation (DMD)** is a technique to train a fast, single-step or few-step generator (the "student") to mimic the output of a slow, multi-step diffusion model (the "teacher"). The goal is to match the output *distribution* of the teacher, not just individual outputs.

**Self Forcing** is a specific application of DMD to video generation. It trains a fast frame-AR model by using a pre-trained, high-quality bidirectional video diffusion model as the teacher. This allows the student model to generate high-quality video in real-time (e.g., in just 4 denoising steps). This paper builds directly on the `Self Forcing` training paradigm.

## 3.2. Previous Works

### 3.2.1. Bidirectional Attention Video Generation
*   **Examples:** `Wan2.1`, `VideoCrafter1`
*   **How it works:** These models process an entire video clip at once. The attention mechanism is "bidirectional," meaning that when generating any given frame, the model can see all other frames in the clip (both past and future).
*   **Pros:** This full context leads to very high-quality and coherent short videos.
*   **Cons:** Computationally very expensive, slow, and cannot be used for streaming or generating videos longer than their fixed training window (typically a few seconds). The attention complexity is quadratic with respect to the number of frames. The formula for the core attention mechanism is:
    \$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    \$
    where `Q, K, V` are the Query, Key, and Value matrices, and $d_k$ is the dimension of the key vectors.

### 3.2.2. Autoregressive Video Generation
*   **Examples:** `Self Forcing`, `LongLive`, `CausVid`
*   **How it works:** These models generate video frame-by-frame or chunk-by-chunk, using "causal attention" where each frame can only attend to past frames. This makes them suitable for long video and real-time generation.
*   **Challenge (Exposure Bias):** Early models suffered from error accumulation. Small errors in early frames would compound over time, leading to degraded quality in long videos. `Self Forcing` largely solved this by ensuring consistency between the training and inference processes.
*   **Challenge (Memory):** As discussed, storing the full `KV cache` for long videos is infeasible. `LongLive` addresses this with a sliding window, which sacrifices long-term memory.

### 3.2.3. Memory Representations for Long-Term Consistency
The paper categorizes previous attempts to solve the long-term consistency problem into three paradigms:
1.  **Explicit 3D Memory:** Methods like `Memory Forcing` convert the 2D video into a 3D point cloud representation. This provides strong geometric consistency but can fail in scenes with poor texture or high dynamics, and depends heavily on the quality of the underlying 3D reconstruction algorithm.
2.  **Implicit 2D Latent Memory:** Methods like `Context as Memory` keep all historical 2D frames and use a retrieval mechanism to select the most relevant ones to condition the next frame generation. This is more flexible than 3D methods but still requires storing all historical frames, leading to high memory consumption.
3.  **Weight Memory:** `TTT-video` proposes updating the model's own weights during inference to "internalize" the memory of past events. While this achieves a constant memory footprint, the need to perform optimization during inference makes it too slow for real-time applications.

## 3.3. Technological Evolution
The field has evolved from:
1.  **High-Quality, Slow, Short Video Models:** Bidirectional attention models (`Wan2.1`) established a high bar for quality but were limited in duration and speed.
2.  **Fast, Real-Time, Short Video Models:** Distillation techniques like DMD and `Self Forcing` enabled frame-AR models to achieve comparable quality in real-time, but still focused on short clips where full history was manageable.
3.  **Real-Time, Long Video Models with Consistency Issues:** To extend to longer videos, methods like `LongLive` introduced memory-saving techniques like sliding windows, but this re-introduced the problem of long-term forgetting.
4.  **The Current Frontier (This Paper):** The goal is now to achieve all three simultaneously: **high quality, real-time speed, and long-term consistency**. MAG tackles this by introducing a sophisticated, learnable memory compression scheme.

## 3.4. Differentiation Analysis
MAG's approach is distinct from previous methods in its core design philosophy:

*   **vs. Window Attention (`LongLive`):** `LongLive` **discards** old information. MAG **compresses** it. This is the fundamental difference that allows MAG to remember scenes from the distant past.
*   **vs. Full History Retention (`Context as Memory`):** `Context as Memory` still requires **storing** every historical frame to be able to retrieve from them. MAG stores only a highly compressed `KV cache`, drastically reducing the memory footprint (e.g., by a factor of 3).
*   **vs. Weight Memory (`TTT-video`):** `TTT-video` encodes memory by changing the model itself, which is slow. MAG's memory mechanism is part of the standard forward pass (generating a `KV cache`), so it incurs no additional computational overhead during inference and remains real-time.
*   **Core Innovation:** The **decoupling** of memory and generation. By training a specialized `memory model`, MAG turns memory compression from a heuristic (like a sliding window) into a learnable and optimizable part of the system.

# 4. Methodology

## 4.1. Principles
The core principle of Memorize-and-Generate (MAG) is to address the conflicting demands of long-term consistency and limited GPU memory by **decoupling memory compression and frame generation**. Instead of using a single, monolithic model that struggles with both tasks, MAG employs two distinct, specialized models:
1.  A **Memory Model** whose sole purpose is to take a sequence of recent frames and compress their essential information into a compact, high-fidelity `KV cache`.
2.  A **Generator Model** that takes this compressed historical `KV cache` as context and synthesizes the next sequence of frames.

    This separation allows for dedicated training objectives for each task, leading to a system that is both memory-efficient and capable of maintaining long-term scene consistency.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Rethinking DMD Optimization in Long Video Generation
The authors first identify a flaw in directly applying the `Self Forcing` (based on DMD) framework to long video generation.

**The Problem: A Degenerate Solution**
The standard DMD training objective aims to make the student generator's output distribution $p_\theta^G(x)$ match the teacher's distribution $p^T(x)$. In long video generation, the generator is conditioned on both history $h$ and a text prompt $T$, so its output is $p_\theta^G(x|h, T)$. However, the powerful pre-trained teacher models (like `Wan2.1`) are typically text-to-video models that only accept a text condition, producing $p^T(x|T)$. The optimization objective thus tries to align $p_\theta^G(x|h, T)$ with $p^T(x|T)$.

The authors argue this creates a shortcut: the model can learn to **ignore the history $h$** and rely solely on the text prompt $T$. Since text and history are often correlated, this can still produce a high-quality video that matches the teacher's output, but it fails to learn the crucial skill of maintaining consistency with the historical context.

**The Solution: History-Focused Loss**
To fix this, they introduce a simple but effective modification. During training, with a certain probability, they force the generator to predict the next video clip with an **empty text condition ($\emptyset$)**. In this case, the generator's output is $p_\theta^G(x|h, \emptyset)$. The model is now forced to rely entirely on the historical context $h$ to generate a plausible continuation that aligns with the teacher's text-conditioned output. This breaks the degenerate solution and compels the model to learn and utilize historical information.

The full training objective combines the original DMD loss and this new history-focused loss. The gradient is formulated as follows:
\$
\nabla_\theta \mathcal{L} = (1 - \lambda) \nabla_\theta \mathcal{L}_{\mathrm{DMD}} + \lambda \nabla_\theta \mathcal{L}_{\mathrm{history}}
\$
where:
*   $\lambda$ is a hyperparameter that balances the two objectives. In practice, this is implemented by randomly choosing to use an empty text prompt with probability $\lambda$.
*   $\nabla_\theta \mathcal{L}_{\mathrm{DMD}}$ is the gradient of the original DMD loss where the generator uses both history and text. The paper provides its approximation from prior work:
    $$
    \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{DMD}} \approx \mathbb{E}_{i \sim U\{1, k\}} \mathbb{E}_{\boldsymbol{z} \sim \mathcal{N}(0, I)} \left[ s^{\mathcal{T}}(\boldsymbol{x}_i) - s_{\boldsymbol{\theta}}^{\mathcal{S}}(\boldsymbol{x}_i) \frac{dG_{\boldsymbol{\theta}}(\boldsymbol{z}_i)}{d\boldsymbol{\theta}} \right]
    $$
    *   $\boldsymbol{\theta}$ represents the parameters of the generator model.
    *   $i \sim U\{1, k\}$ means a clip is uniformly sampled from $k$ segments.
    *   $\boldsymbol{z}$ is Gaussian noise.
    *   $G_{\boldsymbol{\theta}}(\boldsymbol{z}_i)$ is the output of the generator for clip $i$.
    *   $s^{\mathcal{T}}$ and $s_{\boldsymbol{\theta}}^{\mathcal{S}}$ are the score functions of the teacher and student models, respectively. The gradient is essentially pushing the student's score to match the teacher's score.
*   $\nabla_\theta \mathcal{L}_{\mathrm{history}}$ is the gradient of the history-focused loss, which has the same form as $\mathcal{L}_{\mathrm{DMD}}$, but the generated sample $\boldsymbol{x}$ is sampled using an empty text condition: $\boldsymbol{x} \sim p_{\boldsymbol{\theta}}^{G}(\boldsymbol{x} | h, \boldsymbol{\emptyset})$.

### 4.2.2. The Memorize-and-Generate Framework Training Pipeline
The overall training process is divided into two main stages, as illustrated in Figure 2 from the paper.

![Fig. 2: The training pipeline. The training process of MAG comprises two stages. In the first stage, we train the memory model for the triple compressed KV cache, retaining only one frame within a full attention block. The loss function requires the model to reconstruct the pixels of all frames in the block from the compressed cache. The process utilizes a customized attention mask to achieve efficient parallel training. In the second stage, we train the generator model within the long video DMD training framework to adapt to the compressed cache provided by the frozen memory model.](images/2.jpg)

**Stage 1: Training the Memory Model**
The goal of this stage is to train a model that can compress the information of a multi-frame block into the `KV cache` of a single frame. This is framed as an autoencoder-like task.

1.  **Input:** A block of consecutive clean video frames (e.g., 3 frames).
2.  **Encoding:** The model processes this block of frames using full attention (where all frames attend to each other). During this process, it computes the `KV cache` for all tokens across all frames in the block.
3.  **Compression:** After the forward pass, the `KV cache` for all frames *except the last one* is discarded. The `KV cache` of the final frame in the block becomes the compressed latent representation.
4.  **Decoding:** The model is then tasked with a new job: using *only* this compressed `KV cache` (from the last frame) as context, it must denoise random noise to reconstruct the *entire original block* of frames.
5.  **Loss:** The model is trained using a reconstruction loss (e.g., pixel-wise MSE) between the original frames and the reconstructed ones. This forces the model to learn how to pack all the necessary information to reconstruct the whole block into the single `KV cache` it is allowed to keep.
6.  **Parallel Training:** To make this efficient, they use a custom attention mask (shown in Figure 3) that allows the encoding and decoding steps to be performed in a single forward pass for parallel training.

    The attention mask mechanism is detailed in Figure 3 of the paper.

    ![Fig. 3: The attention mask of memory model training. We achieve efficient parallel training of the encode-decode process by concatenating noise and clean frame sequences. By masking out the KV cache of other frames within the block, the model is forced to compress information into the target cache.](images/3.jpg)
    *该图像是示意图，展示了记忆模型训练中的注意力掩码。通过连接噪声和干净帧序列并遮蔽其他帧的 KV 缓存，模型被迫将信息压缩到目标缓存中。*

By concatenating the noisy inputs (for reconstruction) and the clean frames (for encoding), and carefully masking attention, the model is forced to rely only on the compressed target cache for the reconstruction task.

**Stage 2: Training the Generator Model**
Once the memory model is trained, its weights are frozen. It now serves as a fixed, efficient compressor for historical context.

1.  **Initialization:** The `generator model` is initialized with the weights of the trained `memory model`. This ensures they share a common feature space, stabilizing training.
2.  **Training Loop:** The generator is trained in the long video `Self Forcing` framework, using the modified DMD loss from Section 4.2.1.
3.  **Memory Compression in the Loop:** During the autoregressive generation process in training, whenever a new block of frames is generated, instead of appending its full `KV cache` to the history, it is passed through the frozen `memory model` to obtain the compressed `KV cache`.
4.  **Objective:** The generator learns to produce high-quality, consistent video clips conditioned on the **compressed** historical information provided by the memory model. This adapts it to the compressed memory format it will encounter during inference.

    This two-stage process ensures that the final model is both an expert at generating content and an expert at understanding a compressed representation of the past, without compromising on real-time performance.

# 5. Experimental Setup

## 5.1. Datasets
*   **VPData:** A large-scale dataset containing 390,000 high-quality, real-world videos. It was used to train the `memory model` in Stage 1. Its diversity and quality are crucial for learning a general-purpose compression scheme that works across various scenes.
*   **VidProM:** A dataset of text prompts, which the authors extended using a Large Language Model (LLM). This was used for the text-conditioned training of the `generator model` in Stage 2.
*   **MAG-Bench:** A novel benchmark created by the authors specifically for evaluating historical consistency. It consists of 176 videos that feature camera movements where the camera pans or moves away from a scene and then returns. This setup directly tests a model's ability to remember what was previously in the frame. Figure 4 from the paper shows examples from this benchmark.

    ![Fig. 4: Examples from MAG-Bench. MAG-Bench is a lightweight benchmark comprising 176 videos featuring indoor, outdoor, object, and video game scenes. The benchmark also provides appropriate switch times to guide the model toward correct continuation using a few frames.](images/4.jpg)
    *该图像是来自MAG-Bench的示例，展示了不同场景下的视频生成过程。上方展示了在切换标志下的平移操作，下方展示了缩放操作，反映了历史信息的输入和记忆缓存的使用。*

## 5.2. Evaluation Metrics

### 5.2.1. Metrics for Memory Model (Reconstruction Fidelity)
*   **PSNR (Peak Signal-to-Noise Ratio):**
    *   **Conceptual Definition:** Measures the quality of a reconstructed image or video by comparing it to the original. It quantifies the ratio between the maximum possible power of a signal and the power of corrupting noise that affects its fidelity. Higher PSNR values indicate better reconstruction quality.
    *   **Mathematical Formula:**
        \$
        \text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)
        \$
    *   **Symbol Explanation:**
        *   $\text{MAX}_I$ is the maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image).
        *   $\text{MSE}$ is the Mean Squared Error between the original and reconstructed images.

*   **SSIM (Structural Similarity Index Measure):**
    *   **Conceptual Definition:** Measures the similarity between two images based on human perception. Unlike PSNR, which is based on absolute error, SSIM considers changes in luminance, contrast, and structure. Values range from -1 to 1, where 1 indicates perfect similarity.
    *   **Mathematical Formula:**
        \$
        \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
        \$
    *   **Symbol Explanation:**
        *   $\mu_x, \mu_y$ are the average pixel values of images $x$ and $y$.
        *   $\sigma_x^2, \sigma_y^2$ are the variances.
        *   $\sigma_{xy}$ is the covariance.
        *   $c_1, c_2$ are small constants to stabilize the division.

*   **LPIPS (Learned Perceptual Image Patch Similarity):**
    *   **Conceptual Definition:** Measures the perceptual similarity between two images using a pre-trained deep neural network (e.g., VGG). It is designed to align better with human judgment of image similarity than traditional metrics like PSNR and SSIM. Lower LPIPS values indicate higher perceptual similarity.
    *   **Mathematical Formula:** There is no simple closed-form formula. It is computed by passing two images through a network, extracting feature activations from multiple layers, calculating the difference between these activations, scaling them, and averaging the results.

*   **MSE (Mean Squared Error):**
    *   **Conceptual Definition:** Calculates the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. Lower is better.
    *   **Mathematical Formula:**
        \$
        \text{MSE} = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2
        \$
    *   **Symbol Explanation:**
        *   $I$ and $K$ are the two images being compared, of size $m \times n$.
        *   `I(i,j)` and `K(i,j)` are the pixel values at position `(i,j)`.

### 5.2.2. Metrics for Generator Model (Standard T2V)
*   **VBench / VBench-Long:** Comprehensive benchmark suites for evaluating text-to-video models. They assess performance across multiple dimensions, including:
    *   **Quality:** Visual quality, temporal consistency.
    *   **Semantic:** How well the video aligns with the text prompt.
    *   **Background / Subject Consistency:** How well the background and main subjects are maintained throughout the video.

### 5.2.3. Metrics for Historical Consistency
*   **PSNR, SSIM, LPIPS on MAG-Bench:** These standard metrics are used to compare the generated video with the ground truth "return" segment. To handle minor desynchronization in camera movement speed, they use a **Best Match LPIPS** strategy, where each generated frame is compared against the most perceptually similar frame in the ground truth sequence.

## 5.3. Baselines
The paper compares MAG against several representative models:
*   **`Wan2.1`:** A state-of-the-art bidirectional diffusion model. Serves as a reference for high-quality short video generation. It is slow and not autoregressive.
*   **`SkyReels-V2`:** A non-distilled frame-AR model, representing traditional autoregressive methods that require many denoising steps.
*   **`Self Forcing`:** The foundational DMD-based real-time video generation model. It uses full history and is a direct precursor to this work.
*   **`LongLive`:** A state-of-the-art real-time long video generation model that uses a sliding window for memory management. This is a key competitor for the long-term consistency task.
*   **`CausVid`:** Another fast autoregressive model converted from a bidirectional model.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Text-to-Video Generation Performance
The following are the results from Table 1 of the original paper, comparing models on the 5-second VBench benchmark:

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
<td colspan="7"><b>Multi-step model</b></td>
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
<td colspan="7"><b>Few-step distillation model</b></td>
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
<td><b>MAG</b></td>
<td><b>21.7</b></td>
<td><b>83.52</b></td>
<td><b>84.11</b></td>
<td><b>81.14</b></td>
<td><b>97.44</b></td>
<td><b>97.02</b></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Performance:** MAG achieves the highest throughput at **21.7 FPS**, making it the fastest among the compared real-time models. This is attributed to the compressed `KV cache`, which shortens the sequence length for the attention mechanism.
*   **Quality:** MAG's overall score (83.52) is highly competitive with other state-of-the-art distilled models like `Self Forcing` (83.98) and `LongLive` (83.32).
*   **Consistency:** Notably, MAG achieves the best scores in **Background Consistency (97.44)** and **Subject Consistency (97.02)**. This directly supports the authors' claim that their method's focus on preserving historical information improves consistency, even in short videos.

    The following are the results from Table 2 of the original paper, comparing models on the 30-second VBench-long benchmark:

    <table>
    <thead>
    <tr>
    <th rowspan="2">Model</th>
    <th colspan="5">Vbench scores on 30s ↑</th>
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
    <td>Self Forcing [20]</td>
    <td>82.57</td>
    <td>83.30</td>
    <td>79.68</td>
    <td>97.03</td>
    <td>97.80</td>
    </tr>
    <tr>
    <td>Longlive [48]</td>
    <td>82.69</td>
    <td>83.28</td>
    <td>80.32</td>
    <td>97.21</td>
    <td>98.36</td>
    </tr>
    <tr>
    <td><b>MAG</b></td>
    <td><b>82.85</b></td>
    <td><b>83.30</b></td>
    <td><b>81.04</b></td>
    <td><b>97.99</b></td>
    <td><b>99.18</b></td>
    </tr>
    </tbody>
    </table>

**Analysis:**
The trend continues in long video generation. MAG achieves the highest total score and again leads significantly in **Background (97.99)** and **Subject (99.18)** consistency, demonstrating the effectiveness of its memory mechanism over extended durations.

The qualitative comparison in Figure 6 visually supports these numbers, showing MAG produces high-quality and coherent videos.

![Fig. 6: Qualitative comparison on T2V tasks. We present 5-second and 30-second video clips sampled from VBench \[21\] and VBench-Long \[58\], respectively. All methods utilize identical prompts and random initialization noise.](images/6.jpg)
*该图像是一个比较不同方法在T2V任务中生成视频的示意图，展示了5秒和30秒的视频片段，分别采样自VBench和VBench-Long。所有方法均使用相同的提示和随机初始化噪声。*

### 6.1.2. Historical Consistency Performance
This is the central claim of the paper, evaluated on the new `MAG-Bench`.
The following are the results from Table 3 of the original paper:

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
<td><b>MAG</b></td>
<td><b>18.99</b></td>
<td><b>0.60</b></td>
<td><b>0.23</b></td>
<td><b>20.77</b></td>
<td><b>0.66</b></td>
<td><b>0.17</b></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Dominant Performance:** MAG **significantly outperforms** all baselines across all metrics on `MAG-Bench`. For instance, in the more challenging "History Context" setting (where the model generates based on its own past predictions), MAG achieves a PSNR of 18.99, compared to 16.42 for the next best method (`LongLive`). The LPIPS score of 0.23 is also substantially lower (better) than `LongLive`'s 0.32.
*   **Importance of Memory Compression:** `LongLive`, which uses a sliding window, performs better than `Self Forcing` and `CausVid`, which retain full history but weren't explicitly trained to use it effectively. However, MAG's learnable compression is clearly superior to `LongLive`'s naive windowing.
*   **Qualitative Evidence:** Figure 7 provides striking visual proof. When the camera returns to a scene, `Self Forcing` and `LongLive` hallucinate incorrect details (highlighted in red boxes), demonstrating catastrophic forgetting. In contrast, MAG correctly reconstructs the original scene, showing its memory mechanism is working as intended.

    ![Fig. 7: Qualitative comparison on MAG-Bench. We primarily display the visual results of comparable distilled models. Prior to these frames, the models receive and memorize historical frames. Red boxes highlight instances of scene forgetting and hallucinations exhibited by other methods.](images/7.jpg)
    *该图像是图表，展示了MAG框架与其他方法在MAG-Bench上的定性比较。上方为真实图像（GT），中间为MAG模型生成的结果，底部为Self Forcing与Longlive的结果。红框标出其他方法的场景遗忘与幻觉现象。*

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Memory Model Compression Rate
The authors tested how reconstruction quality is affected by the compression rate of the memory model. The rate is equal to the number of frames in a block.
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>Rates</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
<th>MSE×10<sup>2</sup> ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>block=1</td>
<td>34.81</td>
<td>0.93</td>
<td>0.025</td>
<td>0.08</td>
</tr>
<tr>
<td><b>block=3</b></td>
<td><b>31.73</b></td>
<td><b>0.90</b></td>
<td><b>0.045</b></td>
<td><b>0.56</b></td>
</tr>
<tr>
<td>block=4</td>
<td>29.89</td>
<td>0.88</td>
<td>0.059</td>
<td>1.28</td>
</tr>
<tr>
<td>block=5</td>
<td>28.64</td>
<td>0.86</td>
<td>0.071</td>
<td>1.96</td>
</tr>
</tbody>
</table>

**Analysis:**
*   As expected, reconstruction quality decreases as the compression rate increases.
*   The chosen compression rate of **3x** (block=3) provides a strong balance. The PSNR (31.73) and SSIM (0.90) are still very high, indicating excellent reconstruction fidelity. Visual inspection in Figure 5 confirms that the reconstruction is "near-lossless".
*   This study validates their design choice and suggests that even higher compression rates could be explored for applications where memory is extremely constrained.

    ![Fig. 5: Visualization of Memory Model reconstruction results. We display two examples featuring texture detail variations and significant camera movement. Visually, the trained Memory Model achieves near-lossless reconstruction of the original pixels under a $3 \\times$ compression setting.](images/5.jpg)
    *该图像是图示，展示了真实场景（Ground truth）与我们提出的MAG模型生成的结果对比。图中包含两组示例，分别显示了不同场景中的花卉细节变化和相机运动，MAG模型在压缩设置为`3 imes`的情况下，能够实现几乎无损的重建效果。*

### 6.2.2. Necessity of the Memory Model Training (Stage 1)
To prove that their learned compression is better than a naive approach, they ran an ablation ("w/o stage 1" in Table 3) where the learned memory model was replaced with simple 3x downsampling of the `KV cache`.

**Analysis (from Table 3):**
The "w/o stage 1" model performs significantly worse than the full MAG model on the historical consistency task (e.g., PSNR of 17.19 vs. 18.99). It performs similarly to `LongLive`. This demonstrates that simply reducing the cache size is not enough; the **learned compression** from Stage 1 is crucial for preserving the fine-grained details necessary for accurate scene reconstruction and long-term consistency.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully identifies and addresses a critical bottleneck in long-form video generation: the trade-off between long-term consistency and computational feasibility. The proposed **Memorize-and-Generate (MAG)** framework provides an elegant and effective solution by decoupling memory compression from frame generation.

The key findings are:
*   A specialized `memory model` can learn to compress historical information into a compact `KV cache` with high fidelity.
*   Modifying the training objective to include text-free generation forces the `generator model` to effectively utilize this historical context.
*   The resulting MAG system significantly outperforms existing methods in maintaining historical scene consistency, as rigorously measured by the newly introduced `MAG-Bench`.
*   This is achieved without sacrificing performance on standard text-to-video benchmarks and while maintaining real-time inference speeds.

## 7.2. Limitations & Future Work
The authors acknowledge two primary limitations:
1.  **Suboptimal Use of History:** While MAG ensures the fidelity of the stored memory, the `generator model` may not yet know how to optimally *use* this vast amount of historical information. The lack of large-scale datasets specifically designed to train context utilization remains a challenge.
2.  **Dependence on Teacher Model:** The entire framework relies on a powerful, pre-trained teacher model for DMD-based training. This makes it difficult to extend the approach to new domains or tasks, such as action-conditioned world models, without first investing substantial resources into training a new teacher.

    Future work will aim to address these challenges, potentially by curating new datasets for context-aware generation and exploring training paradigms that are less reliant on a pre-trained teacher.

## 7.3. Personal Insights & Critique
*   **Elegant Engineering Solution:** The decoupling of memory and generation is a very strong and practical idea. It isolates a complex problem (memory management) into a self-contained, optimizable module. This design pattern is highly valuable and could be applied to other long-sequence generation tasks beyond video, such as story generation or long-form dialogue.
*   **The Importance of Benchmarking:** The creation of `MAG-Bench` is a significant contribution in itself. Progress in machine learning is often driven by the availability of high-quality benchmarks that can accurately measure the specific capability one aims to improve. `MAG-Bench` provides a much-needed tool for the community to rigorously evaluate long-term consistency.
*   **Potential for Improvement in Memory Objective:** The memory model is trained on a pixel-reconstruction loss. While effective, this may not be optimal. A potential area for improvement could be to train the memory model using a perceptual or feature-space loss. This might encourage the model to prioritize preserving semantically important information over pixel-perfect details, which could lead to even more robust and efficient memory representations.
*   **A Step Towards World Models:** The paper frequently mentions "world models," which aim to build an internal simulation of the world. Long-term consistency is a fundamental prerequisite for such models. MAG represents a significant and practical step in this direction by providing a mechanism for the model to maintain a stable and consistent representation of its environment over time. The real-time nature of the model makes it particularly promising for interactive applications.