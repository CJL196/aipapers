# 1. Bibliographic Information

## 1.1. Title
SemanticGen: Video Generation in Semantic Space

## 1.2. Authors
The authors are Jianhong Bai, Xiaoshi Wu, Xintao Wang, Xiao Fu, Yuanxing Zhang, Qinghe Wang, Xiaoyu Shi, Menghan Xia, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Kun Gai. The affiliations listed are Zhejiang University, Kling Team at Kuaishou Technology, The Chinese University of Hong Kong (CUHK), Dalian University of Technology (DLUT), and Huazhong University of Science and Technology (HUST). This indicates a collaboration between a major Chinese university and a leading tech company's research team, suggesting a blend of academic rigor and industry-level engineering resources.

## 1.3. Journal/Conference
The paper is available on arXiv, which is a preprint server for academic papers. The listed publication date is in the future (December 2025), indicating that this is an early release of the research, likely intended for submission to a top-tier computer vision or machine learning conference such as CVPR, ICCV, or NeurIPS. arXiv is the standard platform for researchers in these fields to share their work quickly with the community.

## 1.4. Publication Year
The paper specifies a future publication date of December 23, 2025. It was submitted to arXiv and is available as a preprint.

## 1.5. Abstract
The abstract introduces `SemanticGen`, a novel video generation model designed to address the slow convergence and high computational cost of existing state-of-the-art models, especially for long videos. Current methods typically operate in the VAE latent space. `SemanticGen` proposes a different approach: generating videos in a compact, high-level semantic space first. The core idea is that videos contain significant redundancy, so global planning should precede the addition of fine details. The method uses a two-stage process: a diffusion model first generates compact semantic video features for the global layout, and a second diffusion model then generates VAE latents conditioned on these features to produce the final video. The authors find that generating in this semantic space leads to faster convergence and is more computationally efficient for long videos. Experimental results show that `SemanticGen` outperforms existing methods and strong baselines in producing high-quality short and long videos.

## 1.6. Original Source Link
*   **Original Source (arXiv):** https://arxiv.org/abs/2512.20619
*   **PDF Link:** https://arxiv.org/pdf/2512.20619v3.pdf
*   **Publication Status:** This is a preprint and has not yet been peer-reviewed or officially published in a conference or journal.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem this paper addresses is the inefficiency and scalability issues in state-of-the-art video generation models. While diffusion-based models have achieved remarkable quality, their training and inference processes are plagued by two major challenges:

1.  **Slow Convergence and High Computational Cost:** Training high-quality video diffusion models requires massive computational resources, often on the scale of hundreds of thousands of GPU-hours. This makes research and development prohibitively expensive and slow.
2.  **Difficulty in Generating Long Videos:** The standard approach involves a diffusion process on video latents produced by a VAE. However, even with VAE compression, a long video (e.g., 60 seconds) can translate into over half a million tokens. Applying bi-directional self-attention, which has quadratic complexity with respect to the number of tokens, becomes computationally infeasible. Existing solutions like sparse attention or autoregressive methods often compromise video quality or suffer from temporal drift (e.g., a character's appearance changing over time).

    The paper's innovative entry point is based on the insight that videos are highly redundant. Instead of modeling a massive set of low-level VAE tokens directly, the authors propose a hierarchical generation process. They argue that generation should begin in a much more compact, high-level **semantic space** where global planning (e.g., scene layout, object motion) can be done efficiently. Once this global plan is established, high-frequency details (e.g., textures, lighting) can be added in a subsequent stage.

## 2.2. Main Contributions / Findings
The paper's primary contributions are:

*   **Proposal of `SemanticGen`:** A novel two-stage video generation framework that first generates a compact semantic representation of a video and then maps it to the detailed VAE latent space. This hierarchical approach separates global planning from detail rendering.
*   **Semantic Representation Compression:** The authors found that using raw, high-dimensional semantic features directly can hinder performance. They introduce a lightweight MLP to compress these features, making the training process more efficient and stable. This compression also regularizes the semantic space to be more amenable to generation by a diffusion model.
*   **Demonstrated Faster Convergence:** A key finding is that modeling in the proposed semantic space leads to significantly faster convergence compared to modeling in a compressed VAE latent space. This directly addresses the computational cost issue of training video models.
*   **Efficient Long Video Generation:** The framework naturally extends to long videos. By performing full attention only in the highly compressed semantic space, it maintains long-term consistency. For the more detailed VAE latent generation, it uses efficient shifted-window attention to avoid quadratic complexity, enabling the generation of minute-long videos without significant quality degradation or drift.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Variational Autoencoder (VAE)
A **Variational Autoencoder (VAE)** is a type of generative neural network used for learning a compressed representation (a latent space) of data. It consists of two main parts:
*   **Encoder:** This network takes an input (e.g., an image or video frame) and compresses it into a low-dimensional latent vector. Unlike a standard autoencoder, a VAE's encoder outputs parameters (mean and variance) for a probability distribution (typically Gaussian) in the latent space. A latent vector is then sampled from this distribution.
*   **Decoder:** This network takes a latent vector and reconstructs the original input from it.
    In the context of this paper, a 3D VAE is used to compress entire video clips into a compact spatio-temporal latent representation. This is crucial for making the subsequent diffusion process computationally manageable.

### 3.1.2. Diffusion Models
**Diffusion Models** are a class of generative models that learn to create data by reversing a gradual noising process. The process involves two parts:
*   **Forward Process:** This is a fixed process where a small amount of Gaussian noise is added to the data in a series of steps. After many steps, the data is transformed into pure, unstructured noise.
*   **Reverse Process (Denoising):** The model, typically a U-Net or Transformer, is trained to reverse this process. At each step, it predicts the noise that was added and subtracts it, gradually transforming pure noise back into a coherent data sample.
    In this paper, diffusion models are the core generative engines, used first to generate semantic features and then to generate VAE latents.

### 3.1.3. Latent Diffusion Models (LDMs)
**Latent Diffusion Models (LDMs)** combine the efficiency of VAEs with the power of diffusion models. Instead of running the computationally expensive diffusion process directly in the high-dimensional pixel space, LDMs first use a pre-trained VAE to encode the data into a much smaller latent space. The diffusion process then operates entirely within this latent space. Once the denoising is complete, the final latent representation is passed through the VAE's decoder to generate the high-resolution output. This is the standard paradigm for most modern high-quality image and video generation models.

### 3.1.4. Transformers and Attention Mechanism
**Transformers** are a neural network architecture that relies heavily on the **self-attention mechanism**. This mechanism allows the model to weigh the importance of different parts of the input sequence when processing a specific part. For video, this means the model can consider information from all other frames and spatial locations when generating a particular patch of a particular frame.
*   **Bi-directional Attention:** All tokens in the sequence can attend to all other tokens, allowing for a global understanding of the entire video. However, its computational and memory cost grows quadratically with the sequence length ($O(N^2)$ for a sequence of length $N$), making it impractical for long videos.
*   **Cross-Attention:** This is a variant where one sequence attends to another. In text-to-video models, the video tokens use cross-attention to attend to the text prompt tokens, ensuring the generated video aligns with the description.

### 3.1.5. Rectified Flow
**Rectified Flow** is a generative modeling framework that formulates generation as a transport problem between a noise distribution and a data distribution. It models this transport along straight paths, which can be described by an Ordinary Differential Equation (ODE). Compared to traditional diffusion models, Rectified Flow often offers benefits like more stable training, faster sampling (fewer steps needed for generation), and a simpler mathematical formulation. `SemanticGen` uses this framework for its diffusion process.

## 3.2. Previous Works
The paper positions itself within the landscape of video generation and the use of semantic representations.

*   **Diffusion-based Video Models:** These models, such as those by Ho et al. (2022) and scaled up with Transformers (e.g., `DiT`), generate all frames simultaneously using bi-directional attention. They produce high-quality short videos but struggle to scale to long videos due to the quadratic complexity of attention.
*   **Autoregressive Video Models:** These models generate video frames or patches sequentially, one after another. This makes them naturally suited for long video generation. However, they can suffer from **temporal drift**, where errors accumulate over time, leading to inconsistencies in later parts of the video.
*   **Hybrid Diffusion-Autoregressive Models:** Works like `Diffusion-forcing` and `Self-forcing` try to combine the global coherence of diffusion with the scalability of autoregressive models. They often involve complex training schemes and may still not match the quality of pure diffusion models.
*   **Semantic Representation in Generation:** The paper notes two main lines of prior work in this area:
    1.  **Improving the VAE:** Some methods (`VA-VAE`, `DC-AE`) modify the VAE training process to make its latent space more "semantic-rich," for instance, by aligning VAE latents with features from pre-trained vision models like CLIP. This is orthogonal to `SemanticGen`'s contribution.
    2.  **Optimizing the Latent Generator:** Other methods operate on the generator itself. `RCG` proposed a two-stage process for *images*: first generating self-supervised representations, then mapping them to pixels. `REPA` aligns the internal states of a diffusion model with semantic features to speed up convergence. The most closely related work is `TokensGen`, which also uses a two-stage pipeline for video.

## 3.3. Technological Evolution
The field has evolved from early GAN-based video models to VAE-based autoregressive models, and now to the dominant paradigm of Latent Diffusion Models. The introduction of the Transformer architecture (`DiT`) allowed these models to scale significantly, leading to breakthroughs in quality (`Sora`, `Kling`). However, this scaling exposed the limitations of full attention for long sequences. `SemanticGen` represents the next logical step in this evolution: instead of trying to make the existing latent space modeling more efficient (e.g., with sparse attention), it proposes changing the space itself to a more abstract and compact semantic one, tackling the efficiency problem at a more fundamental level.

## 3.4. Differentiation Analysis
`SemanticGen`'s core innovation lies in its hierarchical generation strategy centered on a **semantic space** rather than a VAE latent space.

*   **vs. Standard LDMs:** Standard LDMs operate in a single, low-level latent space (the VAE space). `SemanticGen` introduces a two-stage process, with the first stage happening in a much higher-level, more compressed semantic space. This separates global planning from detail rendering.
*   **vs. Autoregressive Models:** `SemanticGen` is not autoregressive. It uses bi-directional attention for global planning, avoiding the temporal drift issue common in purely sequential models. It achieves scalability not by sequential generation, but by using efficient attention mechanisms (`Swin attention`) in the detail-rendering stage.
*   **vs. `TokensGen`:** This is the most critical comparison. `TokensGen` also uses a two-stage approach, but it works by further compressing the **VAE latents**. `SemanticGen` instead uses features from a dedicated **semantic encoder** (a pre-trained video understanding model). The authors empirically show in their ablation study (Figure 9) that generating in a semantic space leads to dramatically faster convergence than generating in a compressed VAE space, suggesting that the semantic space is inherently easier for a diffusion model to learn and sample from.

# 4. Methodology

## 4.1. Principles
The core principle of `SemanticGen` is **hierarchical generation**. The model posits that creating a video is analogous to painting: an artist first sketches the overall composition and shapes (global planning) and then fills in the colors and fine textures (detail rendering). `SemanticGen` mimics this by first generating a video's blueprint in a compact, high-level semantic space, which captures the essence of the content—objects, their layout, and their motion. Only after this blueprint is established does a second model translate it into a high-fidelity video by adding the necessary visual details in the VAE latent space.

The overall architecture is shown in the figure below (Figure 3 from the original paper). It consists of two training stages and one inference stage.

![该图像是示意图，展示了SemanticGen的训练和推理流程，包括两个阶段的生成过程以及输入文本与视频之间的关系。在训练阶段1，VAE潜在生成器生成语义表示；在训练阶段2，语义生成器基于文本生成视频特征；推理阶段则利用VAE潜在生成器最终生成视频。](images/3.jpg)
*该图像是示意图，展示了SemanticGen的训练和推理流程，包括两个阶段的生成过程以及输入文本与视频之间的关系。在训练阶段1，VAE潜在生成器生成语义表示；在训练阶段2，语义生成器基于文本生成视频特征；推理阶段则利用VAE潜在生成器最终生成视频。*

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Preliminary: The Base Text-to-Video Model
`SemanticGen` is built upon a pre-trained text-to-video foundation model. This base model is a standard Latent Diffusion Model that uses a 3D VAE and a Transformer-based diffusion network (`DiT`). It uses the **Rectified Flow** framework for the diffusion and denoising process.

The forward noising process is defined as a straight-line interpolation between the clean data $z_0$ (VAE latents) and pure noise $\epsilon \sim \mathcal{N}(0, I)$:

\$
z _ { t } = ( 1 - t ) z _ { 0 } + t \epsilon
\$

where:
*   $z_0$ is the clean VAE latent representation of the video.
*   $\epsilon$ is a noise vector sampled from a standard normal distribution.
*   $t \in [0, 1]$ is the timestep, where $t=0$ corresponds to clean data and $t=1$ corresponds to pure noise.

    The reverse process learns a velocity field $v_\Theta(z_t, t)$ to transport the noisy data back to the clean data, which is modeled by an ordinary differential equation (ODE):

\$
d z _ { t } = v _ { \Theta } ( z _ { t } , t ) d t
\$

The model with parameters $\Theta$ is trained using the **Conditional Flow Matching** loss, which minimizes the difference between the predicted velocity $v_\Theta$ and the true velocity $u_t$:

\$
\mathcal { L } _ { L C M } = \mathbb { E } _ { t , p _ { t } ( z , \epsilon ) , p ( \epsilon ) } | | v _ { \Theta } ( z _ { t } , t ) - u _ { t } ( z _ { 0 } | \epsilon ) | | _ { 2 } ^ { 2 }
\$

where $u_t(z_0 | \epsilon)$ is the ground-truth velocity vector on the straight path from $\epsilon$ to $z_0$.

During inference, this ODE is solved using a numerical solver like Euler discretization. Starting from pure noise $z_1$, the model iteratively updates the latent representation:

\$
z _ { t } = z _ { t - 1 } + v _ { \Theta } ( z _ { t - 1 } , t ) * \Delta t
\$

This process is repeated for a set number of steps until $t=0$ is reached, yielding the final denoised latent $z_0$.

### 4.2.2. Stage 1: Training the VAE Latent Generator
The first stage involves fine-tuning the base diffusion model to generate VAE latents $z_0$ conditioned on a compressed semantic representation $z_{sem}$. This process is illustrated in Figure 3a of the paper.

1.  **Semantic Feature Extraction:** For a given training video $V$, it is first passed through a pre-trained semantic encoder. The authors chose the vision tower of **Qwen-2.5-VL**, a large vision-language model, because it meets their criteria: it's trained on video data, produces a compact output, and handles diverse content. This encoder outputs a high-dimensional semantic feature map `z'_{sem}`.

2.  **Semantic Representation Compression:** The authors found that directly using the high-dimensional `z'_{sem}` leads to slow convergence. To address this, they introduce a small, learnable **MLP** that compresses `z'_{sem}` into a lower-dimensional representation $z_{sem}$. This MLP is trained to model the compressed feature space as a Gaussian distribution by outputting a mean and variance. A KL divergence loss is used as a regularizer to encourage this distribution to be close to a standard normal distribution, which makes it easier for the second-stage diffusion model to learn.

3.  **In-Context Conditioning and Training:** The compressed semantic representation $z_{sem}$ is then injected into the diffusion model. This is done via **in-context conditioning**, where the noised VAE latents $z_t$ and the semantic features $z_{sem}$ are concatenated along the token dimension before being fed into the Transformer blocks:
    $z_{input} := [z_t, z_{sem}]$
    The model is then trained with the Rectified Flow loss described earlier, learning to denoise $z_t$ to reconstruct $z_0$ while being guided by $z_{sem}$.

### 4.2.3. Stage 2: Training the Semantic Representation Generator
After the VAE latent generator is trained, the second stage focuses on learning to generate the compressed semantic representations $z_{sem}$ from a text prompt. This process is shown in Figure 3b.

*   A separate, smaller diffusion model (the "semantic generator") is trained for this task.
*   The input to this model is the text prompt, and the target output is the corresponding compressed semantic representation $z_{sem}$ (which was generated and stored during Stage 1).
*   During this stage, the semantic encoder (Qwen-2.5-VL) and the compression MLP are **frozen**. The model only learns the distribution of the compressed semantic features.
*   Because the semantic space is highly compressed and regularized by the MLP to be Gaussian-like, this diffusion model converges very quickly.

### 4.2.4. Inference Pipeline
During inference (Figure 3c), the two trained models are chained together:
1.  A user provides a text prompt.
2.  The **semantic generator** (from Stage 2) takes the text prompt and generates a compressed semantic representation $z_{sem}$.
3.  This generated $z_{sem}$ is then passed as a condition to the **VAE latent generator** (from Stage 1).
4.  The VAE latent generator synthesizes the full VAE latents $z_0$ corresponding to the semantic blueprint.
5.  Finally, the pre-trained VAE decoder maps the latents $z_0$ to the pixel space, producing the final video.

### 4.2.5. Extension to Long Video Generation
The `SemanticGen` framework is particularly effective for long videos. The key is to use different attention strategies in the two stages to balance global consistency and computational efficiency.

*   **Semantic Generation Stage:** Because the semantic representation $z_{sem}$ is extremely compact (e.g., 1/16th the number of tokens of the VAE latents), the semantic generator can afford to use **full bi-directional attention** over the entire video length. This ensures that the global plan is coherent and consistent from beginning to end, preventing temporal drift.
*   **VAE Latent Generation Stage:** This stage operates on the much larger set of VAE tokens. To avoid quadratic complexity, the model uses **shifted-window attention (Swin Attention)**, as illustrated in Figure 5 from the paper. Attention is computed only within local windows of frames. To allow information to propagate across windows, the windows are shifted in alternating Transformer layers. This reduces the complexity to be linear with respect to the video length while still allowing for effective local detail rendering guided by the globally consistent semantic features.

    The following figure (Figure 5 from the original paper) illustrates the Swin-Attention mechanism for long videos.

    ![Figure 5. Implementation of Swin-Attention. When generating long videos, we apply full attention to model the semantic representations and use shifted-window attention \[43\] to map them into the VAE space. Thebluesquares indicate VAE latents, while the yellow squares denote semantic representations.](images/5.jpg)
    *该图像是示意图，展示了Swin-Attention的实现过程。在生成长视频时，图中表明应用全注意力机制来建模语义表示，并使用移位窗口注意力将其映射到VAE空间。蓝色方块表示VAE潜变量，而黄色方块表示语义表示。图中还标注了各个部分的宽度，如$T_w$和$T_w/2$。*

# 5. Experimental Setup

## 5.1. Datasets
The authors used two types of datasets for their experiments:
*   **Short Video Generation:** An **internal text-video pair dataset**. The specifics of this dataset (size, content, source) are not disclosed.
*   **Long Video Generation:** The training data was created by splitting clips from **movies and TV shows** into 60-second segments. These segments were then captioned using an **internal captioner** to create text-video pairs.

    The use of internal, non-public datasets is a limitation for reproducibility, as other researchers cannot directly replicate the results or use the same data for fair comparisons. However, this is a common practice in large-scale industrial research.

## 5.2. Evaluation Metrics
The paper uses standard benchmarks and metrics for video generation evaluation.

*   **VBench / VBench-Long:** These are comprehensive benchmark suites designed to evaluate video generative models across multiple dimensions. The specific metrics reported are:
    *   **Subject Consistency:** Measures if the main subject's appearance remains consistent throughout the video.
    *   **Background Consistency:** Measures if the background remains consistent.
    *   **Temporal Flickering:** Quantifies the amount of unnatural, rapid changes in brightness or color between frames. A higher score means less flickering.
    *   **Motion Smoothness:** Assesses the plausibility and smoothness of object motion.
    *   **Imaging Quality:** A general measure of visual quality, including clarity, artifacts, and realism.
    *   **Aesthetic Quality:** A learned metric that predicts the subjective aesthetic appeal of the video.
*   **Drift Metric ($\Delta_{drift}^M$):** This metric is specifically for long video evaluation and quantifies quality degradation over time.
    *   **Conceptual Definition:** It measures the drop in quality between the beginning and the end of a long video. A lower value indicates less drift and better long-term consistency.
    *   **Mathematical Formula:** The paper defines it as the absolute difference in a given metric $M$ between the first 15% and the last 15% of the video frames. If $M_{start}$ is the metric score on the first segment and $M_{end}$ is the score on the last segment, then:
        \$
        \Delta_{drift}^M = |M_{start} - M_{end}|
        \$
    *   **Symbol Explanation:**
        *   $M$: A specific quality metric (e.g., Imaging Quality).
        *   $M_{start}$: The metric computed on the initial segment of the video.
        *   $M_{end}$: The metric computed on the final segment of the video.

## 5.3. Baselines
The paper compares `SemanticGen` against several state-of-the-art and representative baselines.

*   **Short Video Baselines:**
    *   `Wan2.1-T2V-14B`: A large-scale, state-of-the-art text-to-video model.
    *   `HunyuanVideo`: Another strong text-to-video generation model.
*   **Long Video Baselines:**
    *   `SkyReels-V2`: An open-source long video generation model.
    *   `Self-Forcing`: A hybrid diffusion-autoregressive model designed for long videos.
    *   `LongLive`: A model focused on real-time interactive long video generation.

        Critically, the authors also include two internal baselines to ensure a fair comparison:
*   **`Base-CT`:** Their own base model, continuously trained (`CT`) for the same number of steps using the standard diffusion loss, without the `SemanticGen` framework.
*   **`Base-Swin-CT`:** Their base model adapted with Swin attention and continuously trained, to isolate the benefit of the semantic planning stage from the benefit of just using an efficient attention mechanism.

    These internal baselines are crucial as they control for confounding variables like model architecture, training data, and compute, providing a direct measure of the `SemanticGen` framework's contribution.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Short Video Generation
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>Subject Consistency</th>
<th>Background Consistency</th>
<th>Temporal Flickering</th>
<th>Motion Smoothness</th>
<th>Imaging Quality</th>
<th>Aesthetic Quality</th>
</tr>
</thead>
<tbody>
<tr>
<td>Hunyuan-Video [38]</td>
<td>91.11%</td>
<td>95.32%</td>
<td>97.49%</td>
<td>99.07%</td>
<td>64.23%</td>
<td>62.60%</td>
</tr>
<tr>
<td>Wan2.1-T2V-14B [59]</td>
<td>97.23%</td>
<td>98.28%</td>
<td>98.35%</td>
<td>99.08%</td>
<td>66.63%</td>
<td>65.61%</td>
</tr>
<tr>
<td>Base-CT</td>
<td>96.17%</td>
<td>97.27%</td>
<td>98.07%</td>
<td>99.07%</td>
<td>65.77%</td>
<td>63.97%</td>
</tr>
<tr>
<td>SemanticGen</td>
<td><b>97.79%</b></td>
<td><b>97.68%</b></td>
<td><b>98.47%</b></td>
<td><b>99.17%</b></td>
<td>65.23%</td>
<td>64.60%</td>
</tr>
</tbody>
</table>

**Analysis:** For short video generation, `SemanticGen` achieves performance that is highly competitive with state-of-the-art models like `Wan2.1-T2V-14B` and its own direct baseline, `Base-CT`. It scores highest in consistency, temporal quality, and motion smoothness. While its imaging and aesthetic quality scores are slightly lower than the top baseline, the results demonstrate that the two-stage semantic approach does not compromise quality and can even enhance temporal aspects of the video. The qualitative results in Figure 6 further show `SemanticGen`'s superior ability to follow complex text prompts compared to other models.

### 6.1.2. Long Video Generation
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>Subject Consistency</th>
<th>Background Consistency</th>
<th>Temporal Flickering</th>
<th>Motion Smoothness</th>
<th>Imaging Quality</th>
<th>Aesthetic Quality</th>
<th>$\Delta_{drift}^M$</th>
</tr>
</thead>
<tbody>
<tr>
<td>SkyReels-V2 [10]</td>
<td>93.13%</td>
<td>95.11%</td>
<td>98.41%</td>
<td>99.24%</td>
<td>66.00%</td>
<td>62.17%</td>
<td>9.00%</td>
</tr>
<tr>
<td>Self-Forcing [30]</td>
<td>90.41%</td>
<td>93.42%</td>
<td>98.51%</td>
<td>99.17%</td>
<td>70.23%</td>
<td>62.73%</td>
<td>12.39%</td>
</tr>
<tr>
<td>LongLive [70]</td>
<td>94.77%</td>
<td>95.90%</td>
<td>98.48%</td>
<td>99.21%</td>
<td>70.17%</td>
<td>64.73%</td>
<td>4.08%</td>
</tr>
<tr>
<td>Base-CT-Swin</td>
<td>94.01%</td>
<td>94.84%</td>
<td>98.64%</td>
<td>99.32%</td>
<td>68.15%</td>
<td>61.66%</td>
<td>5.20%</td>
</tr>
<tr>
<td>SemanticGen</td>
<td><b>95.07%</b></td>
<td><b>96.70%</b></td>
<td>98.31%</td>
<td><b>99.55%</b></td>
<td><b>70.47 %</b></td>
<td><b>64.09%</b></td>
<td><b>3.58%</b></td>
</tr>
</tbody>
</table>

**Analysis:** The benefits of `SemanticGen` are much more pronounced in long video generation. It outperforms all baselines, including the strong `Base-CT-Swin` baseline, across almost all metrics. Most notably, it achieves the highest scores in **Subject and Background Consistency** and the lowest (best) score on the **$\Delta_{drift}^M$** metric. This strongly validates the core hypothesis: performing global planning with full attention in the compact semantic space effectively mitigates long-term drift and maintains coherence. The qualitative examples in Figure 7 support this, showing `SemanticGen` avoids the color shifts and background inconsistencies that plague other methods.

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. The Effectiveness of Semantic Space Compression
This study investigates the impact of the MLP-based compression of semantic features. The model was trained with three settings: no compression (using 2048-dim features directly), compression to 64 dimensions, and compression to 8 dimensions.

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>Subject Consistency</th>
<th>Background Consistency</th>
<th>Temporal Flickering</th>
<th>Motion Smoothness</th>
<th>Imaging Quality</th>
<th>Aesthetic Quality</th>
</tr>
</thead>
<tbody>
<tr>
<td>w.o. compression (dim=2048)</td>
<td>96.29%</td>
<td>96.54%</td>
<td>96.39%</td>
<td>99.31%</td>
<td>67.42%</td>
<td>58.88%</td>
</tr>
<tr>
<td>w. compression (dim=64)</td>
<td>97.36%</td>
<td>96.85%</td>
<td>98.23%</td>
<td>98.34%</td>
<td>68.16%</td>
<td>60.62%</td>
</tr>
<tr>
<td>w. compression (dim=8)</td>
<td><b>97.49%</b></td>
<td><b>97.34%</b></td>
<td><b>98.27%</b></td>
<td><b>99.38%</b></td>
<td><b>68.43%</b></td>
<td><b>60.95%</b></td>
</tr>
</tbody>
</table>

**Analysis:** The results clearly show that compressing the semantic space significantly improves performance across all metrics. The model with 8-dimensional features performs the best. This confirms the authors' hypothesis that the raw semantic space is not optimal for generation and that compressing and regularizing it makes it easier for the diffusion model to learn, leading to faster convergence and higher-quality results. The qualitative results in Figure 8 visually support this, showing fewer artifacts and better coherence with lower-dimensional features.

### 6.2.2. SemanticGen Achieves Faster Convergence Speed
This crucial ablation compares `SemanticGen`'s approach (modeling in semantic space) to the `TokensGen` approach (modeling in a compressed VAE latent space). Both models were trained from scratch for an equal number of steps (10K).

The results are shown visually in Figure 9.

![该图像是示意图，展示了VAE空间与语义空间中的视频生成过程的对比。在上部分，描绘了一名穿黑色背心的肌肉男子与一名听讲的蓝衣男子的互动；下部分呈现自然风光的变化。该研究的重点在于语义空间中生成视频的有效性。](images/9.jpg)
*该图像是示意图，展示了VAE空间与语义空间中的视频生成过程的对比。在上部分，描绘了一名穿黑色背心的肌肉男子与一名听讲的蓝衣男子的互动；下部分呈现自然风光的变化。该研究的重点在于语义空间中生成视频的有效性。*

**Analysis:** The visual comparison is striking. The model trained in the **semantic space** is already producing coherent and recognizable videos. In contrast, the model trained in the **compressed VAE space** for the same duration only generates coarse, abstract color patches, indicating it is far from converging. This provides powerful evidence for the paper's central claim: the semantic space is fundamentally more "generable" and allows for much faster convergence of the generative model. This is likely because the semantic space abstracts away low-level pixel noise and texture details, allowing the model to focus on learning the core structure and dynamics of the video content first.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully introduces `SemanticGen`, a novel and effective framework for video generation that addresses key challenges of convergence speed and scalability. By proposing a two-stage process that first generates a global plan in a compact semantic space before rendering details, the model achieves significant efficiency gains. The core contributions are the hierarchical generation paradigm, the technique for compressing semantic representations for more effective training, and the demonstration of its powerful application to long video generation. The extensive experiments and strong ablation studies validate that this approach not only accelerates training but also produces high-quality videos that outperform state-of-the-art methods, particularly in maintaining long-term consistency.

## 7.2. Limitations & Future Work
The authors candidly acknowledge several limitations:

*   **Loss of Fine-Grained Details:** Because the global plan is based on semantic features, which abstract away fine details, the model can struggle to maintain perfect consistency of small objects or complex textures over long durations.
*   **Inherited Encoder Limitations:** The performance of `SemanticGen` is inherently tied to the capabilities of its semantic encoder. The chosen encoder (Qwen-2.5-VL) samples video at a low frame rate (1.6 fps), which means it can miss high-frequency temporal events like lightning flashes. The generated video then fails to reproduce these events, as shown in the failure cases.

    Based on these limitations, the authors suggest two main directions for future work:
1.  **Systematic Analysis of Semantic Encoders:** Exploring different types of video understanding models (e.g., self-supervised vs. vision-language) as the semantic encoder to see how their training paradigms affect generation quality.
2.  **Developing Better Semantic Encoders:** Creating new video encoders designed specifically for generation, which would ideally combine high temporal compression with a high sampling rate to capture both long-term structure and high-frequency motion.

## 7.3. Personal Insights & Critique
`SemanticGen` presents a compelling and intuitive solution to a major bottleneck in video generation.

*   **Strengths:**
    *   The core idea of separating global planning from local rendering is elegant and well-motivated. It aligns with how humans create and perceive complex information.
    *   The ablation studies are exceptionally strong, particularly the direct comparison against generating in a compressed VAE space (Figure 9). This provides convincing evidence for the central thesis of the paper.
    *   The solution for long video generation is practical and effective, elegantly combining full attention for global consistency with efficient local attention for detail, which is a significant engineering contribution.

*   **Potential Issues and Areas for Improvement:**
    *   **Reproducibility:** The heavy reliance on internal datasets, models, and captioners is a significant weakness. It makes it impossible for the broader research community to verify the results or build directly upon the work.
    *   **The "Semantic" Bottleneck:** The paper demonstrates the power of semantic space but also reveals its limitations. The quality of the final video is capped by the richness of the semantic representation. If the encoder misses a detail, the generator can never create it. This suggests a potential need for a feedback loop or a mechanism to allow the detail-renderer to request more information from the global planner.
    *   **Generalization:** While the model performs well on movie/TV show data, its ability to generate videos with highly novel or unusual physics/dynamics might be limited by the knowledge contained within the pre-trained semantic encoder. A truly general world simulator might require a semantic space that is learned end-to-end with the generative process rather than being fixed.

        Overall, `SemanticGen` is a high-quality paper that makes a significant contribution to the field. Its core idea of hierarchical generation in semantic space is likely to influence future research in efficient and scalable video synthesis.