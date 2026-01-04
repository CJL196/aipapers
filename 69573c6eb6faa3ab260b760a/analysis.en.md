# 1. Bibliographic Information

## 1.1. Title
Pretraining Frame Preservation in Autoregressive Video Memory Compression

## 1.2. Authors
Lvmin Zhang, Shengqu Cai, Muyang Li, Chong Zeng, Beijia Lu, Anyi Rao, Song Han, Gordon Wetzstein, and Maneesh Agrawala. 
Affiliations include Stanford University, MIT, Carnegie Mellon University, and HKUST.

## 1.3. Journal/Conference
This paper is a preprint currently hosted on arXiv (arXiv:2512.23851). Given the authors' affiliations and the quality of the work, it is likely targeted for a major computer vision or machine learning conference (e.g., CVPR, ICCV, or NeurIPS).

## 1.4. Publication Year
Published on December 28, 2025 (UTC).

## 1.5. Abstract
The paper introduces a neural network architecture designed to compress long video sequences into short, manageable contexts for `autoregressive video generation`. The core innovation is a pretraining objective focused on "frame preservation," which ensures that high-frequency details from any random frame in the history can be retrieved. By first pretraining a memory encoder on this retrieval task, the authors can then fine-tune it as a memory component for `Diffusion Transformers (DiTs)`. This approach allows models to process 20-second histories using only about 5k tokens, maintaining high consistency and perceptual quality while significantly reducing computational costs.

## 1.6. Original Source Link
*   **ArXiv:** [https://arxiv.org/abs/2512.23851](https://arxiv.org/abs/2512.23851)
*   **PDF Link:** [https://arxiv.org/pdf/2512.23851.pdf](https://arxiv.org/pdf/2512.23851.pdf)
*   **Code Repository:** [https://github.com/lllyasviel/PFP](https://github.com/lllyasviel/PFP)

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
Generating long, coherent videos (e.g., movies or long-form stories) is a significant challenge in AI. Most current models use an `autoregressive` approach, where the model predicts the next chunk of video based on the previous history. However, there is a fundamental trade-off: 
1.  **Context Length:** Keeping every frame in memory is computationally impossible due to GPU memory limits (the `quadratic complexity` of attention).
2.  **Context Quality:** Compressing the history (e.g., by skipping frames or downsampling) often results in the loss of fine details, leading to "drifting" where characters' faces or clothes change over time.

    The researchers identified a gap: existing compression methods do not explicitly prioritize the preservation of fine details across the entire temporal span. They aimed to create a "white-box" compression model where the ability to reconstruct any past frame serves as a direct proxy for the quality of the memory.

## 2.2. Main Contributions / Findings
*   **Frame Retrieval Pretraining:** Proposed a new pretraining task where a memory encoder must compress a 20-second video so that any random frame can be reconstructed with high fidelity.
*   **Lightweight Memory Encoder:** Developed an architecture using 3D convolutions and attention that bypasses typical bottlenecks (like VAE channel limits) to output directly into the `latent space` of the generator.
*   **Efficient Autoregressive System:** Demonstrated that a pretrained memory encoder allows for fine-tuning long-video models with much lower computational overhead.
*   **Practical Scaling:** Achieved the compression of 20 seconds of video into a ~5k context length, enabling long-history processing on consumer-grade hardware (like an RTX 4070).

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Autoregressive Video Generation
In this paradigm, a video is generated sequentially. If we want to generate the next segment of video $X_{next}$, we condition the model on all previous segments $H$ (History). 
Mathematically: $P(X_{next} | H)$. 

### 3.1.2. Diffusion Transformers (DiT)
A `Diffusion Transformer` is a model that generates images or videos by starting with pure noise and gradually "denoising" it into a clear image. Unlike older models that used 2D grids (CNNs), DiTs treat video patches as "tokens" in a sequence, similar to how ChatGPT treats words.

### 3.1.3. Latent Space & VAEs
Raw video pixels are massive. Most models first use a `Variational Autoencoder (VAE)` to compress pixels into a smaller, abstract "latent" representation. For example, a $480 \times 832$ image might be compressed into a $30 \times 52$ latent grid. The DiT works in this "latent space" to save memory.

### 3.1.4. Rectified Flow Matching
This is a specific training framework for diffusion models. It learns a straight-line path to transform noise $\epsilon$ into a clean image $X_0$. The noisy version at time $t$ is calculated as:
$X_t = (1 - t)X_0 + t\epsilon$
where $t \in [0, 1]$ is the timestep.

## 3.2. Previous Works
*   **Sliding Windows:** Simple methods that only look at the most recent frames (e.g., the last 2 seconds) and forget everything else. This causes long-term inconsistency.
*   **Token Merging (ToMe):** Methods that combine similar tokens to reduce context length but often blur fine details like facial features.
*   **FramePack:** An earlier method by some of the same authors that "packs" frames into a fixed size, but this paper argues it loses too much "high-frequency" (fine-grained) detail.

## 3.3. Differentiation Analysis
The core difference is the **Pretraining Objective**. While others train the compressor and the generator simultaneously, this paper argues for an independent "pretraining for retrieval" phase. This ensures the compressor is "detail-aware" before it ever tries to help generate new frames.

---

# 4. Methodology

## 4.1. Principles
The intuition is simple: **If a compressed representation contains enough information to reconstruct a specific frame from the past, it definitely contains enough information to maintain consistency in the future.** 

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Stage 1: Pretraining for Frame Retrieval
The goal is to train a compression function $\phi(\cdot)$ that takes a long history $H$ and turns it into a compact context $\phi(H)$. To test this, we try to retrieve a specific subset of frames $\Omega$ using the compressed context.

The following figure (Figure 2 from the original paper) illustrates this pretraining process:

![Figure 2. Pretraining of memory compression models. The memory compression model has to compress long videos (e.g., 20 seconds) into short contexts (e.g., of length 5k). The objective of the pretraining is to retrieve frames with high-frequency details in arbitrary history time positions.](images/2.jpg)

**The Procedural Steps for Pretraining:**
1.  **Sample History:** Take a 20-second video $H$.
2.  **Compress:** Pass $H$ through the encoder $\phi$ to get the compressed tokens $\phi(H)$.
3.  **Random Masking:** Select a random set of frame indices $\Omega$. We keep these frames clean and mask the rest with noise.
4.  **Reconstruction Objective:** The model tries to denoise the noisy frames at indices $\Omega$ using the information stored in $\phi(H)$. The loss function is:
    `L = \mathbb{E}_{H, \Omega, c, \epsilon, t_i} || (\epsilon - H_{\Omega}) - G_{\theta} ((H_{\Omega})_{t_i}, t_i, c, \phi(H)) ||_2^2`
    *   $H_{\Omega}$: The original clean frames at selected positions.
    *   $\epsilon$: Random Gaussian noise.
    *   $G_{\theta}$: A trainable video diffusion model (e.g., Wan or HunyuanVideo).
    *   $t_i$: The diffusion timestep.
    *   $\phi(H)$: The compressed context providing the "memory."

### 4.2.2. Network Architecture
The encoder $\phi$ is designed to be lightweight. It uses a dual-branch approach:
1.  **Low-Resolution (LR) Branch:** Processes a downsampled version of the video to capture global motion and scene structure.
2.  **High-Resolution (HR) Branch:** Processes the original resolution to extract "residual enhancing vectors" (fine details like textures).

    The architecture is shown in the following figure (Figure 3 from the original paper):

    ![Figure 3. Architecture of memory compression model. We use 3D convolution, SiLU, and attention to establish a lightweight neural structure as the baseline compression model. Different alternative architectures (e.g., various channels, full transformer, etc.) are possible and will be discussed in ablation.](images/3.jpg)

    **Layer Breakdown:**
*   **3D Convolutions:** Used to reduce the spatial and temporal dimensions. For example, a $4 \times 4 \times 2$ rate means the width and height are reduced by 4x, and the time (frames) by 2x.
*   **Feature Projection:** Instead of going through the narrow 16-channel VAE bottleneck, the encoder outputs directly into the DiT's inner channel dimension (e.g., 3072 or 5120 channels), preserving more data.

### 4.2.3. Stage 2: Fine-tuning for Autoregressive Generation
Once the encoder $\phi$ is pretrained, it is frozen or fine-tuned alongside the main video generator to produce *new* frames.

The following figure (Figure 4 from the original paper) shows the transition to fine-tuning:

![Figure 4. Finetuning autoregressive video models. We illustrate the finetuning and inference of the final autoregressive video models. The pretraining of the memory compression model is finished before the finetuning.](images/4.jpg)

**The Generation Flow:**
1.  **Input:** Current noise $X_t$, text prompt $c$, and the compressed history $\phi(H)$.
2.  **Diffusion Step:** The model predicts the clean version of the next frame $X_0$:
    $L_{FT} = \mathbb{E}_{X_0, H, c, \epsilon, t_i} || (\epsilon - X_0) - G_{\theta} (X_{t_i}, t_i, c, \phi(H)) ||_2^2$
3.  **Iterative Concatenation:** The newly generated frames are added to the history $H$, which is then re-compressed by $\phi(H)$ for the next step.

    ---

# 5. Experimental Setup

## 5.1. Datasets
*   **Size:** 5 million internet videos.
*   **Content:** A mix of horizontal (widescreen) and vertical (Shorts-style) videos.
*   **Annotations:** Captioned using `Gemini-2.5-flash` in a "storyboard" format (descriptions with timestamps). 
*   **Example:** A video of a grandmother petting a cat would have captions like: "0s: Woman stands by shelf," "12s: Woman pets cat," "22s: Woman sits down."

## 5.2. Evaluation Metrics

### 5.2.1. PSNR (Peak Signal-to-Noise Ratio)
1.  **Conceptual Definition:** Measures the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher is better.
2.  **Mathematical Formula:**
    `PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)`
3.  **Symbol Explanation:** $MAX_I$ is the maximum pixel value (e.g., 255); `MSE` is the Mean Squared Error between the original and reconstructed frame.

### 5.2.2. SSIM (Structural Similarity Index Measure)
1.  **Conceptual Definition:** Evaluates the perceived change in structural information between two images.
2.  **Mathematical Formula:**
    $SSIM(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$
3.  **Symbol Explanation:** $\mu$ is the mean; $\sigma^2$ is the variance; $\sigma_{xy}$ is the covariance; $c_1, c_2$ are constants to stabilize the division.

### 5.2.3. LPIPS (Learned Perceptual Image Patch Similarity)
1.  **Conceptual Definition:** Uses a deep neural network to measure how similar two images look to a human. Lower is better.

### 5.2.4. Consistency Metrics (Cloth, Identity, Object)
1.  **Conceptual Definition:** Use Vision-Language Models (VLMs) like `Gemini` or `LLaVA` to answer questions: "Is the character wearing the same shirt as in the previous scene?"

## 5.3. Baselines
*   **Large Patchifier:** Increases the patch size of the DiT (equivalent to `FramePack`).
*   **Only LR:** Only uses the low-resolution branch of the encoder.
*   **Without Pretrain:** Trains the system from scratch without the frame-retrieval pretraining phase.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experiments confirm that pretraining is the "secret sauce." Models without pretraining often "hallucinate" new details that don't match the history, while the `Proposed` method maintains consistent character identity and background details even after 20 seconds.

The following are the results from Table 1 of the original paper, showing the reconstruction quality during the pretraining phase:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :--- | :--- |
| Large Patchifier* (4×4×2) | 12.93 | 0.412 | 0.365 |
| Only LR (4×4×2) | 15.21 | 0.472 | 0.212 |
| Without LR (4×4×2) | 15.73 | 0.423 | 0.198 |
| **Proposed (4×4×2)** | **17.41** | **0.596** | **0.171** |
| Proposed (2×2×2) | 19.12 | 0.683 | 0.152 |
| Proposed (2×2×1) | 20.19 | 0.705 | 0.121 |

**Analysis:** The "Proposed (4×4×2)" significantly outperforms the "Large Patchifier" (FramePack) across all metrics, proving that the dual-branch architecture and retrieval objective are superior at preserving details.

The following are the results from Table 2 of the original paper, evaluating the final video consistency:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">Human</th>
<th rowspan="2">Object ↑</th>
<th rowspan="2">User Study ELO ↑</th>
</tr>
<tr>
<th>Cloth ↑</th>
<th>Identity ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>WanI2V + QwenEdit (2p)</td>
<td>95.09</td>
<td>68.22</td>
<td>91.19</td>
<td>1198</td>
</tr>
<tr>
<td>Only LR (4×4×2)</td>
<td>91.98</td>
<td>69.22</td>
<td>85.32</td>
<td>1194</td>
</tr>
<tr>
<td>Without Pretrain (4×4×2)</td>
<td>87.12</td>
<td>66.99</td>
<td>81.13</td>
<td>N/A</td>
</tr>
<tr>
<td>**Proposed (4×4×2)**</td>
<td>**96.12**</td>
<td>**70.73**</td>
<td>**89.89**</td>
<td>**1216**</td>
</tr>
<tr>
<td>**Proposed (2×2×2)**</td>
<td>**96.71**</td>
<td>**72.12**</td>
<td>**90.27**</td>
<td>**1218**</td>
</tr>
</tbody>
</table>

**Analysis:** The proposed method achieves higher ELO scores and better identity/cloth consistency than existing baselines. The difference between "Without Pretrain" (87.12) and "Proposed" (96.12) in cloth consistency is a stark validation of the methodology.

## 6.2. Visual Comparison
The effect of pretraining is visually apparent. The following figure (Figure 6 from the original paper) shows how the model with pretraining correctly remembers facial features and clothing styles, whereas the model without it creates inconsistent characters:

![该图像是一个示意图，展示了预训练在自回归视频记忆压缩中的应用。图中分为三组：历史（20秒）、使用预训练（建议）和未使用预训练，分别展示了不同情况下的图像特征对比。](images/6.jpg)
*该图像是一个示意图，展示了预训练在自回归视频记忆压缩中的应用。图中分为三组：历史（20秒）、使用预训练（建议）和未使用预训练，分别展示了不同情况下的图像特征对比。*

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that **frame preservation** is a highly effective objective for video memory compression. By training a model to "remember and retrieve" past frames, it naturally learns to encode the specific details (faces, textures, lighting) required for long-term consistency in video generation. This allows for a massive reduction in context length (down to 5k tokens for 20s of video) without sacrificing the "storytelling" quality.

## 7.2. Limitations & Future Work
*   **Error Accumulation (Drifting):** While improved, the model still faces "drifting" in very long shots (single continuous takes without cuts). The authors suggest that specialized training on "single-shot continuation" is still needed.
*   **Computational Cost:** Pretraining on 5 million videos requires significant resources ($8 \times \text{H100}$ cluster), even if the final inference is efficient.
*   **Multi-Modal Integration:** Future work could explore integrating audio or more complex storyboard instructions into the compression space.

## 7.3. Personal Insights & Critique
**Innovation:** The move from "implicit memory" (just training on next-frame prediction) to "explicit retrieval" is a brilliant move towards more explainable AI. It allows researchers to verify if the memory works *before* running the expensive generation phase.

**Application:** This technology is a massive win for the "indie creator" community. Being able to run long-context video generation on a 12GB GPU (RTX 4070) democratizes high-quality video storytelling, which was previously the domain of companies with massive server farms.

**Critique:** The paper relies heavily on VLMs (Gemini) for evaluation. While VLMs are getting better, they can have their own biases. Supplementing this with more traditional geometric or optical flow metrics for all experiments would have added another layer of rigor.