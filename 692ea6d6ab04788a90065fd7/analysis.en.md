# 1. Bibliographic Information
## 1.1. Title
CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer

## 1.2. Authors
Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Fengt, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, Jie Tang.
The authors are affiliated with Tsinghua University and Zhipu AI. This suggests a collaboration between academia and industry in cutting-edge AI research.

## 1.3. Journal/Conference
The paper is published as a preprint on arXiv (arXiv:2408.06072). While not yet peer-reviewed for a specific journal or conference, arXiv is a widely recognized platform for disseminating early research in computer science and other fields, allowing for rapid sharing and feedback within the scientific community.

## 1.4. Publication Year
2024

## 1.5. Abstract
This paper introduces `CogVideoX`, a large-scale text-to-video generation model built upon a diffusion transformer architecture. It is capable of generating 10-second continuous videos at 16 frames per second (fps) with a resolution of $768 \times 1360$ pixels, aligning precisely with text prompts. The authors address common limitations of previous video generation models, such as limited movement, short durations, and difficulty in generating coherent narratives. They propose several novel designs: a `3D Variational Autoencoder (VAE)` for improved spatial and temporal video compression and fidelity; an `expert transformer` with `expert adaptive LayerNorm` to enhance deep fusion between text and video modalities; and a progressive training strategy combined with `multi-resolution frame pack` technique to enable coherent, long-duration, dynamic video generation with various aspect ratios. Furthermore, an effective `text-video data processing pipeline`, including advanced video captioning, contributes significantly to generation quality and semantic alignment. Experimental results demonstrate `CogVideoX` achieves state-of-the-art performance across automated benchmarks and human evaluations. The model weights for the `3D Causal VAE`, `Video caption model`, and `CogVideoX` are publicly released.

## 1.6. Original Source Link
The official source link is https://arxiv.org/abs/2408.06072. The PDF link is https://arxiv.org/pdf/2408.06072v3.pdf. It is currently a preprint (version 3).

# 2. Executive Summary
## 2.1. Background & Motivation
The field of text-to-video generation has seen rapid advancements, driven by Transformer architectures and diffusion models. Notable progress has been made with `Diffusion Transformers (DiT)` as demonstrated by models like Sora. However, despite these advancements, current video generation models still face significant challenges:
*   **Limited Movement and Short Durations:** Many existing models struggle to generate videos with substantial motion and are often restricted to very short durations (e.g., 2-3 seconds).
*   **Lack of Coherent Narratives:** Generating videos that tell a consistent story or depict complex, long-term actions based on a text prompt remains difficult (e.g., "a bolt of lightning splits a rock, and a person jumps out from inside the rock"). This implies issues with temporal consistency and capturing large-scale motions.
*   **Computational Cost:** High-dimensional video data requires immense computational resources, making efficient compression and modeling crucial.
*   **Data Quality and Annotation:** Online video data often lacks accurate and comprehensive textual descriptions, which is vital for training text-to-video models.

    The core problem the paper aims to solve is enabling the generation of `long-duration`, `temporally consistent` videos with `rich motion semantics` and `high resolution` from text prompts, while addressing the underlying computational and data challenges.

## 2.2. Main Contributions / Findings
The paper introduces `CogVideoX` and makes several key contributions:
*   **Novel Architecture for Coherent, Long-Duration, High-Action Video Generation:** `CogVideoX` integrates a `3D causal VAE` and an `expert transformer` to generate videos up to 10 seconds long, at 16fps, $768 \times 1360$ resolution, and with multiple aspect ratios. This design specifically targets generating coherent videos with significant motion, which was a major limitation of prior work.
*   **Efficient Video Compression with 3D Causal VAE:** The introduction of a `3D VAE` that compresses videos along both spatial and temporal dimensions significantly improves compression rates and video fidelity, while also preventing flicker and ensuring continuity between frames. This is a crucial step for handling high-dimensional video data efficiently.
*   **Enhanced Text-Video Alignment via Expert Transformer:** The `expert transformer` equipped with `expert adaptive LayerNorm` facilitates a deeper and more effective fusion between text and video modalities. This design improves the model's ability to align generated video content with the given text prompts. The use of `3D full attention` further ensures temporal consistency and captures large-scale motions.
*   **Robust Training Strategies:**
    *   **Progressive Training:** A multi-stage progressive training approach, including `multi-resolution frame pack` and resolution-progressive training, is employed to effectively utilize diverse data and learn knowledge at different resolutions, enhancing generation performance and stability.
    *   **Explicit Uniform Sampling:** A novel sampling method for diffusion timesteps is proposed, which stabilizes the training loss curve and accelerates convergence.
*   **Effective Data Processing Pipeline:** A comprehensive `text-video data processing pipeline`, including advanced video filtering and an innovative `video captioning model` (`CogVLM2-Caption`), is developed. This pipeline generates high-quality textual descriptions for training data, significantly improving the model's semantic understanding.
*   **State-of-the-Art Performance:** `CogVideoX` demonstrates state-of-the-art performance in both automated benchmarks (e.g., `FVD`, `CLIP4Clip`, `Dynamic Quality`, `GPT4o-MTScore`) and human evaluations against openly-accessible top-performing models and even competitive closed-source models like `Kling`.
*   **Public Release:** The authors publicly release the `5B` and `2B` models, including text-to-video and image-to-video versions, along with the `3D Causal VAE` and `Video caption model`, positioning them as commercial-grade open-source video generation models.

    These contributions collectively address the critical challenges in text-to-video generation, enabling the creation of more dynamic, coherent, and longer videos at higher resolutions.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
To understand `CogVideoX`, familiarity with several core concepts in deep learning and generative AI is essential:

*   **Transformers (Vaswani et al., 2017):** The Transformer is an neural network architecture that relies on self-attention mechanisms, allowing it to weigh the importance of different parts of the input sequence when processing each element. It revolutionized natural language processing (NLP) and has since been adapted for computer vision.
    *   **Self-Attention:** A mechanism that allows the model to assess the relationships between different positions in a sequence to compute a representation of the sequence. For each token in the input, it computes an output that is a weighted sum of all input tokens, where the weights are determined by a compatibility function (e.g., dot product) between the query vector of the current token and the key vectors of all tokens.
    *   **Multi-Head Attention:** An extension of self-attention where the attention mechanism is run multiple times in parallel, each with different learned linear projections of the queries, keys, and values. The outputs are then concatenated and linearly transformed.
    *   **Layer Normalization (`LayerNorm`):** A technique used to normalize the activations of a layer across the features of an input, making training faster and more stable by reducing internal covariate shift.
    *   **Positional Encoding:** Since Transformers are permutation-invariant (they don't inherently understand the order of tokens), positional encodings are added to the input embeddings to inject information about the relative or absolute position of tokens in the sequence. `Rotary Position Embedding (RoPE)` is a relative positional encoding method that applies a rotation matrix to query and key vectors, effectively encoding relative position information.

*   **Diffusion Models (Ho et al., 2020):** A class of generative models that learn to reverse a gradual diffusion process. They start with a simple noise distribution, and through a series of steps, gradually transform this noise into a data sample (e.g., an image or video).
    *   **Forward Diffusion Process:** Gradually adds Gaussian noise to an input data sample over $T$ timesteps, eventually transforming it into pure noise.
    *   **Reverse Diffusion Process:** A neural network (often a `U-Net` or `Transformer`) is trained to predict and remove the noise at each timestep, effectively reversing the forward process and generating data from noise.
    *   **Latent Diffusion Models (LDM):** Diffusion models that operate in a compressed `latent space` rather than directly on pixel space. This significantly reduces computational cost. `Variational Autoencoders (VAEs)` are commonly used to encode data into and decode from this latent space.
    *   **v-prediction (Salimans & Ho, 2022):** An alternative prediction objective for diffusion models where the model predicts a $v$ vector (a reparameterization of the noise and the original data) instead of directly predicting the noise or the original data. This often leads to improved sample quality and faster convergence.
    *   **zero SNR (Lin et al., 2024):** Refers to a specific noise schedule or parameterization in diffusion models, often related to how the signal-to-noise ratio (SNR) is handled at the boundaries of the diffusion process.

*   **Variational Autoencoders (VAEs):** A type of generative model that learns a compressed, continuous `latent representation` of data.
    *   **Encoder:** Maps input data to a distribution (mean and variance) in the latent space.
    *   **Decoder:** Reconstructs data samples from the latent space.
    *   **Kullback-Leibler (KL) Divergence:** A measure of how one probability distribution diverges from another. In VAEs, it regularizes the latent space by forcing the encoded distribution to be close to a prior distribution (e.g., a standard Gaussian).
    *   **3D VAE:** An extension of VAEs to handle 3D data (e.g., video data with spatial and temporal dimensions) by using 3D convolutions in its encoder and decoder.

*   **Generative Adversarial Networks (GANs):** A class of generative models consisting of two neural networks: a generator that creates synthetic data, and a discriminator that distinguishes between real and synthetic data. They are trained in an adversarial manner. In some generative models, a `GAN loss` might be incorporated to refine the output quality.

*   **FlashAttention (Dao et al., 2022):** An optimized attention algorithm that speeds up Transformer training and inference by reducing memory access costs, particularly for long sequences. It achieves this by reordering operations and performing computations in blocks, exploiting memory hierarchy.

## 3.2. Previous Works
The paper contextualizes `CogVideoX` against a backdrop of evolving text-to-video generation techniques:

*   **Early Transformer-based Text-to-Video Models:**
    *   `CogVideo (Hong et al., 2022)`: An early attempt to pretrain and scale Transformers for video generation from text.
    *   `Phenaki (Villegas et al., 2022)`: Another promising model that explored the use of Transformers for variable-length video generation from text.
    *   These models demonstrated the potential of Transformers but often faced challenges with `limited motion` and `short durations`.

*   **Diffusion Models for Video Generation:**
    *   `Make-A-Video (Singer et al., 2022)`: One of the pioneering works applying diffusion models to text-to-video generation.
    *   `Imagen Video (Ho et al., 2022)`: Another significant advancement, generating high-definition videos with diffusion models.
    *   These models showed impressive capabilities but often relied on `cascading multiple super-resolution and frame interpolation models` to achieve longer or higher-resolution videos, leading to `limited semantic information` and `minimal motion` in the base generation.
    *   `Stable Video Diffusion (SVD) (Blattmann et al., 2023)`: Scaled latent video diffusion models to large datasets. It tried to fine-tune a 2D image VAE decoder to address jittering issues. However, `CogVideoX` notes this approach doesn't fully exploit `temporal redundancy` and might not achieve optimal compression.
    *   `AnimateDiff (Guo et al., 2023)`: Focused on animating personalized text-to-image diffusion models.

*   **Diffusion Transformers (DiT):**
    *   `DiT (Peebles & Xie, 2023)`: Using Transformers as the backbone of diffusion models, this architecture marked a new milestone, exemplified by `OpenAI's Sora (OpenAI, 2024)`. `CogVideoX` explicitly builds upon the success of DiT.

*   **Video VAEs:**
    *   Earlier video models often directly used 2D image VAEs. `CogVideoX` points out that this only models spatial dimensions, leading to `jittery videos`.
    *   Recent models like `Open-Sora (Zheng et al., 2024b)` and `Open-Sora-Plan (PKU-Yuan Lab & Tuzhan AI etc., 2024)` have started using `3D VAE` for temporal compression, but `CogVideoX` argues they might still suffer from `blurry` and `jittery videos` due to small latent channels.

## 3.3. Technological Evolution
The evolution of text-to-video generation can be traced through:
1.  **Early GAN-based methods:** Focused on basic video generation, often with limited coherence or resolution.
2.  **Autoregressive models (e.g., VideoGPT):** Generated videos frame by frame or token by token, often struggling with long-range consistency.
3.  **Transformer-based autoregressive models (e.g., CogVideo, Phenaki):** Extended Transformers to video, improving coherence over longer durations but still constrained by computational cost and motion generation.
4.  **Diffusion Models (e.g., Make-A-Video, Imagen Video):** Achieved significant breakthroughs in visual quality and realism by denoising a latent representation, but often required cascading steps for high-res/long videos.
5.  **Latent Diffusion Models with 2D VAEs:** Improved efficiency by operating in latent space, but temporal consistency remained a challenge (e.g., SVD's fine-tuning of 2D VAE).
6.  **Diffusion Transformers (DiT):** Combined the scalability of Transformers with the generative power of diffusion, setting new benchmarks (e.g., Sora).
7.  **3D VAEs for Video:** Recent trend to directly compress temporal information, addressing jitter, but `CogVideoX` identifies limitations in existing 3D VAE designs.

    `CogVideoX` positions itself at step 7, building on the `DiT` architecture and advancing `3D VAE` and `attention mechanisms` specifically for video.

## 3.4. Differentiation Analysis
`CogVideoX` differentiates itself from previous `DiT`-based and video generation models through several key innovations:

*   **Dedicated 3D Causal VAE:** Unlike `SVD` which fine-tunes a 2D VAE or other models using 3D VAEs with small latent channels (e.g., `Open-Sora`), `CogVideoX` designs and trains a specialized `3D causal VAE` with optimized compression ratios and latent channels. This allows for higher compression and significantly better video fidelity and temporal continuity, directly addressing `flickering` and `jitter` issues.
*   **Expert Transformer with Expert Adaptive LayerNorm:** Instead of separate transformers for text and video (`MMDiT`) or simple concatenation, `CogVideoX` introduces `Expert Adaptive LayerNorm`. This mechanism allows for deep fusion and alignment of modalities within a single transformer, optimizing parameter usage and simplifying the architecture while outperforming alternatives.
*   **3D Full Attention:** Many previous works (`Make-A-Video`, `AnimateDiff`) used `separated spatial and temporal attention` to reduce complexity. `CogVideoX` argues this approach struggles with `large movements` and `consistency`. By employing `3D full attention` and leveraging `FlashAttention` for efficiency, `CogVideoX` directly models both spatial and temporal dimensions comprehensively, leading to better `temporal consistency` and `motion capture`.
*   **Multi-Resolution Frame Pack & Progressive Training:** While some models address varied resolutions using bucketing (`SDXL`), `CogVideoX` introduces `Multi-Resolution Frame Pack` for mixed-duration and mixed-resolution training within a batch. Combined with `progressive training` across resolutions, this approach maximizes data utilization and model robustness for diverse video shapes, which is a more sophisticated handling of varied input data.
*   **Explicit Uniform Sampling:** This novel sampling strategy for diffusion timesteps directly tackles loss fluctuation and training instability observed in standard uniform sampling, offering a practical improvement to the diffusion training process.
*   **Advanced Data Pipeline with Custom Captioning:** Recognizing the limitations of existing video captions, `CogVideoX` develops a `Dense Video Caption Data Generation pipeline` using `GPT-4` and fine-tuned `LLaMA2` (and later `CogVLM2-Caption`). This custom pipeline generates significantly more detailed and accurate video descriptions, which is crucial for training high-quality text-to-video models, providing a superior data foundation compared to relying solely on existing, often short, captions.

    In essence, `CogVideoX` focuses on architectural innovations for `temporal coherence` and `motion` (3D VAE, 3D Full Attention), `modality alignment` (Expert Adaptive LayerNorm), and `training stability/efficiency` (Progressive Training, Explicit Uniform Sampling, advanced data pipeline), setting it apart from models that might excel in image quality but falter in video dynamics or rely on simpler architectural choices.

# 4. Methodology
## 4.1. Principles
The core idea behind `CogVideoX` is to build a `Diffusion Transformer (DiT)` that is highly efficient and effective for generating long, coherent, and dynamic videos from text prompts. It achieves this by addressing three main challenges:
1.  **High-dimensional video data compression:** Videos are inherently high-dimensional. To make them computationally tractable for a diffusion model, efficient compression that preserves both spatial and temporal information is crucial. The paper proposes a `3D Variational Autoencoder (VAE)` to achieve this.
2.  **Deep fusion of text and video modalities:** Accurate alignment between the input text prompt and the generated video is paramount. The model needs a mechanism to effectively integrate textual semantics into the visual generation process. This is tackled by an `expert transformer` with `expert adaptive LayerNorm`.
3.  **Temporal consistency and dynamic motion:** Generating videos with smooth, coherent motion over an extended duration, especially for large movements, is a significant hurdle. This is addressed by using `3D full attention` and robust training strategies like `progressive training` and `multi-resolution frame pack`.

    The overall principle is to combine a specialized video compression scheme with an advanced Transformer architecture and refined training techniques to overcome the limitations of previous text-to-video models.

## 4.2. Core Methodology In-depth (Layer by Layer)
The `CogVideoX` architecture processes text and video inputs through several stages, as illustrated in Figure 3.

![Figure 3: The overall architecture of CogVideoX.](images/3.jpg)  
*该图像是一个示意图，展示了CogVideoX模型中专家变压器的结构与流程，包括文本编码器和3D因果VAE的关联，以及3D全注意力机制的详细设计。*

Figure 3: The overall architecture of CogVideoX.

### 4.2.1. Input Processing
1.  **Video Input:** A video is fed into the `3D Causal VAE`.
    *   **3D Causal VAE Encoder:** This component compresses the raw video data, which contains both spatial and temporal information, into a lower-dimensional `latent space`.
        *   **Structure:** As shown in Figure 4 (a), the `3D VAE` consists of an encoder, a decoder, and a `Kullback-Leibler (KL) regularizer`. The encoder uses symmetrically arranged stages with `3D convolutions` and `ResNet blocks`. Some blocks perform `3D downsampling` (across spatial and temporal dimensions), while others only perform `2D downsampling` (spatial only).
        *   **Temporally Causal Convolution:** To ensure that future frames do not influence the processing of current or past frames, `temporally causal convolutions` are adopted. As depicted in Figure 4 (b), padding is applied only at the beginning of the convolution space.
        *   **Compression Rate:** The `3D VAE` achieves an $8 \times 8 \times 4$ compression ratio, meaning the latent representation is 8 times smaller in height, 8 times smaller in width, and 4 times smaller in the temporal dimension compared to the input video pixels. For example, a $768 \times 1360$ frame at 16fps over 10 seconds (160 frames) would be compressed significantly.
        *   **Output:** The encoder outputs a video latent representation of shape $T \times H \times W \times C$, where $T$ is the number of latent frames, $H$ and $W$ are the latent height and width, and $C$ is the number of latent channels.

2.  **Text Input:** The textual prompt is encoded into `text embeddings` $z_{\mathrm{text}}$ using a `T5` model (Raffel et al., 2020).
    *   **T5 (Text-to-Text Transfer Transformer):** A large language model trained on a massive text dataset. It provides rich semantic representations of text, which are crucial for aligning with visual content.

### 4.2.2. Latent Patchification and Concatenation
1.  **Vision Latent Patchification:** The compressed video latents $z_{\mathrm{vision}}$ (from the `3D VAE` encoder) are `patchified` and `unfolded` into a long sequence of vision tokens.
    *   A patchification operation divides the $T \times H \times W \times C$ latent tensor into smaller, non-overlapping 3D patches (e.g., $q \times p \times p$ where $q$ is temporal patch size and $p$ is spatial patch size). Each patch is then flattened into a token.
    *   The resulting sequence length is $\frac{T}{q} \cdot \frac{H}{p} \cdot \frac{W}{p}$. The paper mentions that when $q > 1$, images at the beginning of the sequence are used to enable joint training of images and videos.

2.  **Modality Concatenation:** The patchified vision tokens ($z_{\mathrm{vision}}$) and the text embeddings ($z_{\mathrm{text}}$) are concatenated along the sequence dimension. This creates a unified input sequence for the `expert transformer`, enabling joint processing of visual and textual information.

### 4.2.3. Expert Transformer
The concatenated embeddings are fed into a stack of `expert transformer blocks`. This is the core diffusion model backbone.
1.  **3D-RoPE (Rotary Position Embedding):** To encode positional information in the video sequence, the `Rotary Position Embedding (RoPE)` (Su et al., 2024) is extended to `3D-RoPE`.
    *   Each latent in the video tensor has a 3D coordinate `(x, y, t)` (spatial x, spatial y, temporal t).
    *   `1D-RoPE` is independently applied to each dimension ($x$, $y$, $t$).
    *   The hidden states' channel is divided: 3/8 for x-dimension, 3/8 for y-dimension, and 2/8 for t-dimension.
    *   The resulting 1D-RoPE encodings for `x, y, t` are then concatenated along the channel dimension to form the final `3D-RoPE` encoding, which is added to the token embeddings. This provides relative positional information essential for video coherence.

2.  **Expert Adaptive LayerNorm (Expert AdaLN):** This component is crucial for handling the numerical and feature space differences between text and video modalities while processing them in the same sequence.
    *   **Modulation:** Following `DiT` (Peebles & Xie, 2023), the `timestep` $t$ of the diffusion process is used as input to a `modulation module`. This module generates scaling and shifting parameters for `LayerNorm`.
    *   **Modality-Specific Normalization:** Instead of a single `LayerNorm`, `CogVideoX` uses two independent `Expert Adaptive LayerNorms`:
        *   `Vision Expert Adaptive LayerNorm (Vision Expert AdaLN)`: Applies the `timestep`-dependent modulation parameters to the vision hidden states.
        *   `Text Expert Adaptive LayerNorm (Text Expert AdaLN)`: Applies different `timestep`-dependent modulation parameters to the text hidden states.
    *   This strategy promotes alignment of feature spaces across the two modalities without significantly increasing parameters, unlike `MMDiT` which uses separate transformers.

3.  **3D Full Attention:** To effectively capture temporal consistency and large-scale motions in videos, `CogVideoX` employs `3D full attention`.
    *   **Challenge of Separated Attention:** As illustrated in Figure 5, previous methods using `separated spatial and temporal attention` (e.g., `Make-A-Video`, `AnimateDiff`) struggle to maintain consistency for `large movements` because visual information between adjacent frames (like a person's head) can only be `implicitly transmitted` through background patches, increasing learning complexity.

        ![Figure 5: The separated spatial and temporal attention makes it challenging to handle the large motion between adjacent frames. In the figure, the head of the person in frame $i + 1$ cannot directly attend to the head in frame $i$ . Instead, visual information can only be implicitly transmitted through other background patches. This can lead to inconsistency issues in the generated videos.](images/5.jpg)
        *该图像是示意图，展示了视频帧之间的注意力机制。左侧是当前帧（frame i）的图像，右侧是下一个帧（frame i+1）。图中用箭头标示了“no attention”（无注意力）和“implicit transmission”（隐式传递）的区分，强调了时间维度的相互影响。右上角的图例说明了不同颜色代表的空间注意力（红色）和时间注意力（黄色）。此图反映了CogVideoX模型在视频生成中的注意力机制设计。*

    Figure 5: The separated spatial and temporal attention makes it challenging to handle the large motion between adjacent frames. In the figure, the head of the person in frame $i + 1$ cannot directly attend to the head in frame $i$ . Instead, visual information can only be implicitly transmitted through other background patches. This can lead to inconsistency issues in the generated videos.

    *   **Solution:** `3D full attention` computes attention across *all* spatial and temporal dimensions simultaneously for each token. This allows direct interaction between any patch in the video sequence, regardless of its spatial or temporal distance, making it more robust to `large motions`.
    *   **Efficiency:** Leveraging optimizations like `FlashAttention` (Dao et al., 2022) allows `3D full attention` to be computationally feasible even for long video sequences, avoiding the computational explosion typically associated with full attention on high-dimensional data.

### 4.2.4. Output Decoding
1.  **Unpatchification:** After processing by the `expert transformer`, the output tokens are `unpatchified` to restore the original latent shape ($T \times H \times W \times C$).
2.  **3D Causal VAE Decoder:** The unpatchified latents are then fed into the `3D Causal VAE decoder`, which reconstructs the final video in pixel space. This decoder mirrors the encoder structure but performs `upsampling`.

### 4.2.5. Training Objectives and Strategies
The model is trained to predict the noise $\epsilon$ in the latent space at a given timestep $t$, based on the noised latent $\tilde{z}$ and the text prompt $z_{\mathrm{text}}$.

1.  **Diffusion Objective:** The training objective follows the `v-prediction` paradigm (Salimans & Ho, 2022) and `zero SNR` (Lin et al., 2024), similar to `LDM` (Rombach et al., 2022). The objective is to minimize:
    $$
    L _ { \mathrm { s i m p l e } } ( \theta ) : = \mathbf { E } _ { t , x _ { 0 } , \epsilon } \big \| \epsilon - \epsilon _ { \theta } \big ( \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , t \big ) \big \| ^ { 2 }
    $$
    Where:
    *   $L_{\mathrm{simple}}(\theta)$: The simplified training loss for the model parameters $\theta$.
    *   $\mathbf{E}_{t, x_0, \epsilon}$: Expectation over timesteps $t$, original data $x_0$, and sampled noise $\epsilon$.
    *   $\epsilon$: The true noise added at timestep $t$.
    *   $\epsilon_{\theta}(\cdot, t)$: The neural network (the `expert transformer` in `CogVideoX`) parameterized by $\theta$, which predicts the noise given the noised latent and the timestep $t$.
    *   $\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$: The noised latent variable at timestep $t$, which is effectively the input to the noise prediction network.
    *   $\bar{\alpha}_t$: A schedule parameter that determines the amount of noise added at timestep $t$.

2.  **Mixed-Duration Training and Multi-Resolution Frame Pack:**
    *   **Problem:** Traditional methods often train with fixed frame counts, leading to data underutilization and generalization issues when mixing images (1 frame) and videos (many frames). Inconsistent data shapes in a batch also complicate training.
    *   **Solution:** Inspired by `Patch'n Pack` (Dehghani et al., 2024), `CogVideoX` uses `mixed-duration training`. Videos of different lengths and resolutions are grouped into the same batch, but their shapes are made consistent using `Multi-Resolution Frame Pack`.
    *   **Illustration:** Figure 6 visualizes this. Instead of padding smaller videos to the maximum length (which wastes computation), `Multi-Resolution Frame Pack` likely refers to a dynamic packing or batching strategy that efficiently groups items of similar sizes or adaptively handles varying sizes within the same batch.
    *   **RoPE Adaptation:** `3D RoPE` is adapted to different resolutions and durations. The paper chooses `extrapolation` over `interpolation` for `RoPE` to maintain relative position clarity, as seen in Figure 9 where extrapolation leads to clearer, repetitive images at higher resolutions.

        ![Figure 6: The diagram of mixed-duration training and Frame Pack. To fully utilize the data and enhance the model's generalization capability, we train on videos of different duration within the same batch.](images/6.jpg)
        *该图像是示意图，展示了传统的图像视频联合训练与提出的多分辨率帧打包方法之间的对比。左侧说明了旧方法中图像与固定长度视频训练任务之间的较大差距，而右侧则展示了通过多分辨率帧打包（Multi-Resolution Frame Pack）以缩小这个差距的方法。图中包含一个示例，展示了不同帧数的视频及其分辨率。*

    Figure 6: The diagram of mixed-duration training and Frame Pack. To fully utilize the data and enhance the model's generalization capability, we train on videos of different duration within the same batch.

    ![Figure 9: The comparison between the initial generation states of extrapolation and interpolation when increasing the resolution with RoPE. Extrapolation tends to generate multiple small, clear, and repetitive images, while interpolation generates a blurry large image.](images/9.jpg)  
    *该图像是示意图，展示了RoPE的外推（Extrapolation）与插值（Interpolation）效果对比。左侧的外推图像清晰可辨，右侧的插值图像则显得模糊，体现了两者在效果上的差异。*

    Figure 9: The comparison between the initial generation states of extrapolation and interpolation when increasing the resolution with RoPE. Extrapolation tends to generate multiple small, clear, and repetitive images, while interpolation generates a blurry large image.

3.  **Progressive Training:**
    *   **Strategy:** The model learns in stages, starting with lower-resolution videos and progressively increasing resolution.
        *   Stage 1: Train on `256px` videos to learn `semantic` and `low-frequency knowledge`.
        *   Subsequent Stages: Gradually increase resolutions to `512px`, then `768px` to learn `high-frequency knowledge`.
    *   **Aspect Ratio Preservation:** The aspect ratio is maintained during resizing, with only the short side scaled.
    *   **High-Quality Fine-Tuning:** A final fine-tuning stage is performed on a subset of `higher-quality video data` (20% of the total dataset) to remove artifacts like subtitles/watermarks and improve visual quality.

4.  **Explicit Uniform Sampling:**
    *   **Problem:** Standard uniform sampling of timesteps $t$ in each data parallel rank can lead to non-uniform distributions in practice and significant fluctuations in loss, as diffusion loss magnitude correlates with timesteps.
    *   **Solution:** `Explicit Uniform Sampling` divides the range `[1, T]` into $n$ intervals, where $n$ is the number of ranks. Each rank then uniformly samples only within its assigned interval.
    *   **Benefit:** This method ensures a more uniform distribution of sampled timesteps across the entire training process, leading to a `more stable loss curve` and `accelerated convergence`, as shown in Figure 10 (d).

5.  **Data Processing Pipeline:**
    *   **Video Filtering:** A set of `negative labels` (e.g., `Editing`, `Lack of Motion Connectivity`, `Low Quality`, `Lecture Type`, `Text Dominated`, `Noisy Screenshots`) are defined.
        *   `Video-LLaMA (Zhang et al., 2023b)` based classifiers are trained on 20,000 manually labeled videos to screen out low-quality data.
        *   `Optical flow scores` and `image aesthetic scores` are calculated and dynamically thresholded to ensure `dynamic` and `aesthetic quality`.
        *   Approximately `35M` high-quality video clips (avg. 6 seconds) remain after filtering.
    *   **Video Captioning:** To address the lack of descriptive captions for most video data, a `Dense Video Caption Data Generation pipeline` is developed (Figure 7).
        *   **Stage 1 (Initial Caption):** Use the `Panda70M (Chen et al., 2024b)` model to generate short video captions.
        *   **Stage 2 (Dense Image Captions):** Extract frames from videos and use the image recaptioning model `CogVLM (Wang et al., 2023a)` (from `CogView3 (Zheng et al., 2024a)`) to create dense image captions for each frame.
        *   **Stage 3 (Summary with GPT-4):** Use `GPT-4` to summarize all the dense image captions into a final, comprehensive video caption.
        *   **Stage 4 (Accelerated Summary):** To scale this process, a `LLaMA2 (Touvron et al., 2023)` model is fine-tuned using the `GPT-4` generated summaries.
        *   **Stage 5 (End-to-End Captioning):** Further fine-tune an end-to-end video understanding model, `CogVLM2-Caption` (based on `CogVLM2-Video (Hong et al., 2024)` and `Llama3 (AI@Meta, 2024)`), using the generated dense caption data for large-scale, detailed video description.

            ![Figure 7: The pipeline for dense video caption data generation. In this pipeline, we generate short video captions with the Panda70M model, extract frames to create dense image captions, and use GPT-4 to summarize these into final video captions. To accelerate this process, we fine-tuned a Llama 2 model with the GPT-4 summaries.](images/7.jpg)
            *该图像是示意图，展示了CogVLM2-Video模型的数据处理流程，包括输入视频的分帧、短视频字幕生成、图像字幕生成和长视频字幕输出。图中标示出不同版本的数据路径，体现了信息流的转化和处理机制。*

    Figure 7: The pipeline for dense video caption data generation. In this pipeline, we generate short video captions with the Panda70M model, extract frames to create dense image captions, and use GPT-4 to summarize these into final video captions. To accelerate this process, we fine-tuned a Llama 2 model with the GPT-4 summaries.

# 5. Experimental Setup
## 5.1. Datasets
`CogVideoX` is trained on a comprehensive dataset constructed from various sources and processed through a custom pipeline.

*   **Primary Video Dataset:** A collection of `approximately 35M high-quality video clips` (averaging about 6 seconds each) with corresponding text descriptions. This dataset is obtained after rigorous filtering using custom video filters.
*   **Image Datasets:** `2 billion images` filtered with aesthetic scores from `LAION-5B (Schuhmann et al., 2022)` and `COYO-700M (Byeon et al., 2022)` datasets are used to assist training. These image datasets provide a rich source of spatial information and aesthetic understanding.
*   **Video Captioning Datasets (for `CogVLM2-Caption` training):**
    *   `Panda70M (Chen et al., 2024b)`: Used to generate initial short video captions.
    *   `COCO Caption (Lin et al., 2014)` and `WebVid (Bain et al., 2021b)`: Mentioned as examples of existing video caption datasets, but their captions are noted to be generally too short for comprehensive video description.
    *   **Custom Dense Video Caption Data:** The `Dense Video Caption Data Generation pipeline` (Figure 7) is crucial here. It leverages `CogVLM` for image captioning of video frames and `GPT-4` (and later fine-tuned `LLaMA2`/`CogVLM2-Caption`) for summarizing these into detailed video captions. This custom dataset is essential for improving semantic alignment.

        **Example of data sample:** The paper provides examples of generated videos based on text prompts such as "A few golden retrievers playing in the snow" (Figures 11, 12) or "Mushroom turns into a bear" (Figure 15). For the captioning models, it shows a "meticulously crafted white dragon with a serene expression and piercing blue eyes..." as a `CogVLM2-Caption` output for a video, indicating the level of detail aimed for in the training captions.

These datasets were chosen to provide a large volume of data, ensure high quality (through filtering), and offer detailed textual descriptions (through custom captioning) necessary for training a robust text-to-video diffusion model capable of generating coherent, long-duration, and semantically aligned videos.

## 5.2. Evaluation Metrics
The evaluation of `CogVideoX` employs a combination of automated metrics and human evaluation, focusing on various aspects of video generation quality and fidelity.

### 5.2.1. Automated Metric Evaluation

The following metrics, mostly sourced from `Vbench (Huang et al., 2024)`, are used:

1.  **Flickering (↓):**
    *   **Conceptual Definition:** This metric quantifies the degree of temporal instability or "jitter" in a video sequence, specifically evaluating frame-to-frame consistency. Lower values indicate smoother, more stable video generation.
    *   **Mathematical Formula:** The paper states it calculates the `L1 difference` between each pair of adjacent frames.
        $$
        \text{Flickering} = \frac{1}{N-1} \sum_{i=1}^{N-1} \|F_i - F_{i+1}\|_1
        $$
    *   **Symbol Explanation:**
        *   $N$: Total number of frames in the video.
        *   $F_i$: The $i$-th frame of the video.
        *   $F_{i+1}$: The $(i+1)$-th frame of the video.
        *   $\|\cdot\|_1$: The `L1 norm` (Manhattan distance), which calculates the sum of the absolute differences between corresponding pixel values of two frames.

2.  **Peak Signal-to-Noise Ratio (PSNR) (↑):**
    *   **Conceptual Definition:** A common metric used to quantify the quality of reconstruction of lossy compression codecs or generative models. It measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR values indicate better reconstruction quality.
    *   **Mathematical Formula:**
        $$
        \text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)
        $$
        where
        $$
        \text{MSE} = \frac{1}{WH} \sum_{x=1}^{W} \sum_{y=1}^{H} (I(x,y) - K(x,y))^2
        $$
    *   **Symbol Explanation:**
        *   $\text{MAX}_I$: The maximum possible pixel value of the image (e.g., 255 for an 8-bit image).
        *   $\text{MSE}$: `Mean Squared Error` between the original and reconstructed images.
        *   $W$: Width of the image.
        *   $H$: Height of the image.
        *   `I(x,y)`: Pixel value of the original image at coordinate `(x,y)`.
        *   `K(x,y)`: Pixel value of the reconstructed image at coordinate `(x,y)`.

3.  **Fréchet Video Distance (FVD) (↓):**
    *   **Conceptual Definition:** A metric used to evaluate the quality of generated videos by measuring the "distance" between the feature distributions of real and generated video sequences. It's an extension of Fréchet Inception Distance (FID) for images. Lower FVD scores indicate higher quality and realism in generated videos.
    *   **Mathematical Formula:** FVD is based on embedding videos into a feature space using a pre-trained neural network (e.g., a 3D Inception network) and then calculating the Fréchet distance between the multivariate Gaussian distributions fitted to these embeddings for real and generated videos.
        $$
        \text{FVD} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
        $$
    *   **Symbol Explanation:**
        *   $\mu_r$: Mean of the feature embeddings for real videos.
        *   $\mu_g$: Mean of the feature embeddings for generated videos.
        *   $\Sigma_r$: Covariance matrix of the feature embeddings for real videos.
        *   $\Sigma_g$: Covariance matrix of the feature embeddings for generated videos.
        *   $\|\cdot\|^2$: Squared Euclidean distance.
        *   $\text{Tr}(\cdot)$: Trace of a matrix.

4.  **CLIP4Clip Score (Luo et al., 2022) (↑):**
    *   **Conceptual Definition:** Measures the semantic alignment between generated videos and their corresponding text prompts. It uses a `CLIP` (Contrastive Language-Image Pre-training) model, adapted for video, to embed both the video and text into a common feature space. The score reflects the cosine similarity between these embeddings. Higher scores indicate better text-video alignment.
    *   **Mathematical Formula:**
        $$
        \text{CLIP4Clip Score} = \text{CosineSimilarity}(E_v, E_t) = \frac{E_v \cdot E_t}{\|E_v\| \|E_t\|}
        $$
    *   **Symbol Explanation:**
        *   $E_v$: Feature embedding of the generated video from the CLIP4Clip video encoder.
        *   $E_t$: Feature embedding of the text prompt from the CLIP4Clip text encoder.
        *   $\text{CosineSimilarity}(\cdot, \cdot)$: The cosine similarity function.

5.  **Vbench Metrics (from Huang et al., 2024) (↑):** These evaluate specific aspects of video content generation. Higher scores are better.
    *   **Human Action:** Measures the realism and coherence of human actions in the video.
    *   **Scene:** Evaluates the quality and consistency of the generated scene.
    *   **Dynamic Degree:** Quantifies the amount and realism of motion and dynamic elements within the video. This is crucial for evaluating videos that are not merely static scenes with minor changes.
    *   **Multiple Objects:** Assesses the model's ability to generate and manage multiple distinct objects coherently.
    *   **Appearance Style:** Evaluates how well the generated video adheres to a specified or implied aesthetic style.

6.  **Dynamic Quality (Liao et al., 2024) (↑):**
    *   **Conceptual Definition:** A composite metric specifically designed to evaluate video generation quality while mitigating biases that arise from negative correlations between video dynamics and static video quality. It integrates various quality metrics with dynamic scores, ensuring that highly dynamic videos are not penalized for being less "still" or simple.
    *   **Mathematical Formula:** Not explicitly provided in the paper, but conceptualized as an integration of quality and dynamics.

7.  **GPT4o-MTScore (Yuan et al., 2024) (↑):**
    *   **Conceptual Definition:** Measures the `metamorphic amplitude` of time-lapse videos using `GPT-4o`. This metric assesses how much meaningful change or transformation occurs over time in the video, relevant for depicting physical, biological, or meteorological changes. It gauges the model's ability to generate videos with significant narrative progression or dynamic evolution.
    *   **Mathematical Formula:** Not explicitly provided, but relies on `GPT-4o` for assessment.

### 5.2.2. Human Evaluation

A comprehensive human evaluation framework is established to assess general capabilities across four aspects:

1.  **Sensory Quality:** Focuses on perceptual quality, including subject consistency, frame continuity, and stability.
    *   **Score 1:** High sensory quality (consistent appearance, high stability, high resolution, realistic composition/color, visually appealing).
    *   **Score 0.5:** Average sensory quality (80% consistent appearance, moderate stability, 50% high resolution, 70% realistic composition/color, some visual appeal).
    *   **Score 0:** Poor sensory quality (large inconsistencies, low resolution, unrealistic composition).

2.  **Instruction Following:** Focuses on alignment with the text prompt, including accuracy of subject, quantity, elements, and details.
    *   **Score 1:** 100% follow text instructions (correct elements, quantity, features).
    *   **Score 0.5:** 100% follow text instructions, but with minor flaws (distorted subjects, inaccurate features).
    *   **Score 0:** Does not 100% follow instructions (inaccurate elements, incorrect quantity, incomplete elements, inaccurate features). *Note: Total score cannot exceed 2 if instructions are not followed.*

3.  **Physics Simulation:** Focuses on adherence to physical laws, such as lighting, object interactions, and fluid dynamics realism.
    *   **Score 1:** Good physical realism (real-time tracking, dynamic realism, realistic lighting/shadow, high interaction fidelity, accurate fluid motion).
    *   **Score 0.5:** Average physical realism (some degradation, unnatural transitions, lighting/shadow mismatches, distorted interactions, floating fluid motion).
    *   **Score 0:** Poor physical realism (results do not match reality, obviously fake).

4.  **Cover Quality:** Focuses on metrics assessable from single-frame images, including aesthetic quality, clarity, and fidelity.
    *   **Score 1:** Image clear, subject obvious, complete display, normal color tone.
    *   **Score 0.5:** Image quality average, subject relatively complete, normal color tone.
    *   **Score 0:** Cover image low resolution, blurry.

## 5.3. Baselines
The paper compares `CogVideoX` against several prominent open-source and closed-source text-to-video models.

### 5.3.1. Baselines for Automated Metric Evaluation
The following models are used as baselines for automated metrics, as shown in Table 3:
*   `T2V-Turbo (Li et al., 2024)`
*   `AnimateDiff (Guo et al., 2023)`
*   `VideoCrafter-2.0 (Chen et al., 2024a)`
*   `OpenSora V1.2 (Zheng et al., 2024b)`
*   `Show-1 (Zhang et al., 2023a)`
*   `Gen-2 (runway, 2023)` (a commercial/closed-source model)
*   `Pika (pik, 2023)` (a commercial/closed-source model)
*   `LaVie-2 (Wang et al., 2023b)`

    These baselines represent a range of state-of-the-art text-to-video generation models, including those based on diffusion models, latent diffusion, and various architectural improvements. They are considered representative for evaluating generation quality, dynamism, and text alignment.

### 5.3.2. Baselines for 3D VAE Comparison
For the `3D VAE` reconstruction quality, `CogVideoX` compares its `3D VAE` with:
*   `Open-Sora`
*   `Open-Sora-Plan`
    These models also utilize `3D VAEs`, making them relevant comparisons for the video compression component.

### 5.3.3. Baselines for Human Evaluation
For human evaluation, `CogVideoX-5B` is compared against:
*   `Kling (2024.7)`: A prominent closed-source model, representing a high benchmark in the field. This comparison is particularly valuable as `Kling` is known for its strong performance.

    The choice of these baselines ensures a comprehensive comparison across open-source and closed-source models, covering different architectural approaches and levels of performance.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The experimental results demonstrate `CogVideoX`'s superior performance across various automated and human evaluation metrics, highlighting its ability to generate high-quality, long-duration, dynamic, and semantically coherent videos.

### 6.1.1. 3D VAE Reconstruction Effect
The `3D VAE` is a critical component for efficient and high-fidelity video compression.
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th></th>
<th>Flickering ↓</th>
<th>PSNR ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>Open-Sora</td>
<td>92.4</td>
<td>28.5</td>
</tr>
<tr>
<td>Open-Sora-Plan</td>
<td>90.2</td>
<td>27.6</td>
</tr>
<tr>
<td>Ours</td>
<td>85.5</td>
<td>29.1</td>
</tr>
</tbody>
</table>

Table 2: Comparison with the performance of other spatiotemporal compression VAEs.

`CogVideoX`'s `3D VAE` ("Ours") achieves the best `PSNR` (29.1) and the lowest `Flickering` score (85.5) compared to `Open-Sora` and `Open-Sora-Plan` on $256 \times 256$ resolution 17-frame videos. This indicates that its `3D VAE` design provides superior reconstruction quality and temporal stability, effectively addressing the `jitter` problem identified in prior works. The note that "other VAE methods use fewer latent channels than ours" suggests that `CogVideoX`'s VAE design (e.g., using 32 latent channels for variant B in Table 1) is better optimized for video fidelity.

### 6.1.2. Automated Metric Evaluation with Text-to-Video Models
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<td rowspan="2">Models</td>
<td rowspan="2">Human Action</td>
<td rowspan="2">Scene</td>
<td colspan="4">Dynamic Multiple Appear. Dynamic GPT4o-MT</td>
<td rowspan="2">Score</td>
</tr>
<tr>
<td>Degree</td>
<td>Objects</td>
<td>Style</td>
<td>Quality</td>
</tr>
</thead>
<tbody>
<tr>
<td>T2V-Turbo(Li et al., 2024)</td>
<td>95.2</td>
<td>55.58</td>
<td>49.17</td>
<td>54.65</td>
<td>24.42</td>
<td></td>
<td></td>
</tr>
<tr>
<td>AnimateDiffGuo et al. (2023)</td>
<td>92.6</td>
<td>50.19</td>
<td>40.83</td>
<td>36.88</td>
<td>22.42</td>
<td></td>
<td>2.62</td>
</tr>
<tr>
<td>VideoCrafter-2.0(Chen et al., 2024a)</td>
<td>95.0</td>
<td>55.29</td>
<td>42.50</td>
<td>40.66</td>
<td>25.13</td>
<td>43.6</td>
<td>2.68</td>
</tr>
<tr>
<td>OpenSora V1.2(Zheng et al., 2024b)</td>
<td>85.8</td>
<td>42.47</td>
<td>47.22</td>
<td>58.41</td>
<td>23.89</td>
<td>63.7</td>
<td>2.52</td>
</tr>
<tr>
<td>Show-1(Zhang et al., 2023a)</td>
<td>95.6</td>
<td>47.03</td>
<td>44.44</td>
<td>45.47</td>
<td>23.06</td>
<td>57.7</td>
<td>−</td>
</tr>
<tr>
<td>Gen-2(runway, 2023)</td>
<td>89.2</td>
<td>48.91</td>
<td>18.89</td>
<td>55.47</td>
<td>19.34</td>
<td>43.6</td>
<td>2.62</td>
</tr>
<tr>
<td>Pika(pik, 2023)</td>
<td>88.0</td>
<td>44.80</td>
<td>37.22</td>
<td>46.69</td>
<td>21.89</td>
<td>52.1</td>
<td>2.48</td>
</tr>
<tr>
<td>LaVie-2(Wang et al., 2023b)</td>
<td>96.4</td>
<td>49.59</td>
<td>31.11</td>
<td>64.88</td>
<td>25.09</td>
<td></td>
<td>2.46</td>
</tr>
<tr>
<td>CogVideoX-2B</td>
<td>96.6</td>
<td>55.35</td>
<td>66.39</td>
<td>57.68</td>
<td>24.37</td>
<td>57.7</td>
<td>3.09</td>
</tr>
<tr>
<td>CogVideoX-5B</td>
<td>96.8</td>
<td>55.44</td>
<td>62.22</td>
<td>70.95</td>
<td>24.44</td>
<td>69.5</td>
<td>3.36</td>
</tr>
</tbody>
</table>

Table 3: Evaluation results of CogVideoX-5B and CogVideoX-2B.

`CogVideoX-5B` demonstrates state-of-the-art performance across the board:
*   It achieves the highest scores in `Human Action` (96.8), `Scene` (55.44), `Dynamic Degree` (62.22), `Multiple Objects` (70.95), and `GPT4o-MTScore` (3.36).
*   It is highly competitive in `Appearance Style` (24.44) and `Dynamic Quality` (69.5).
*   The `CogVideoX-2B` model also shows strong performance, often outperforming many larger baselines, indicating the scalability of the proposed architecture. For instance, `CogVideoX-2B` has a `Dynamic Degree` of 66.39, which is even higher than `CogVideoX-5B`, possibly due to `CogVideoX-5B` trading off some raw dynamism for overall coherence or fine-grained detail.
*   The superior scores in `Dynamic Degree` and `GPT4o-MTScore` are particularly significant, as these metrics directly assess the model's ability to generate videos with rich, coherent motion and meaningful temporal changes, which were explicit goals of the research.

    The radar chart in Figure 2 visually reinforces these performance advantages, showing `CogVideoX-5B` and `CogVideoX-2B` dominating several axes.

    ![Figure 2: The performance of openly-accessible text-to-video models in different aspects.](images/2.jpg)
    *该图像是一个雷达图，展示了不同视频生成模型在各个评估指标上的表现，如动态质量、场景、动作等。CogVideoX-2B和CogVideoX-5B在多个指标上表现优越，特别是在动态质量方面。图中包含各模型的得分信息。*

Figure 2: The performance of openly-accessible text-to-video models in different aspects.

### 6.1.3. Human Evaluation
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<td>Model</td>
<td>Sensory Quality</td>
<td>Instruction Following</td>
<td>Physics Simulation</td>
<td>Cover Quality</td>
<td>Total Score</td>
</tr>
</thead>
<tbody>
<tr>
<td>Kling</td>
<td>0.638</td>
<td>0.367</td>
<td>0.561</td>
<td>0.668</td>
<td>2.17</td>
</tr>
<tr>
<td>CogVideoX-5B</td>
<td>0.722</td>
<td>0.495</td>
<td>0.667</td>
<td>0.712</td>
<td>2.74</td>
</tr>
</tbody>
</table>

Table 4: Human evaluation between CogVideoX and Kling.

In human evaluation, `CogVideoX-5B` consistently outperforms `Kling`, a strong closed-source model:
*   `CogVideoX-5B` achieves higher scores across all four human evaluation aspects: `Sensory Quality` (0.722 vs 0.638), `Instruction Following` (0.495 vs 0.367), `Physics Simulation` (0.667 vs 0.561), and `Cover Quality` (0.712 vs 0.668).
*   The `Total Score` for `CogVideoX-5B` (2.74) is significantly higher than `Kling`'s (2.17).
    This indicates that `CogVideoX` not only excels in objective metrics but also in subjective human perception of video quality, coherence, physical realism, and adherence to prompts. The improvement in `Instruction Following` is particularly noteworthy, suggesting the effectiveness of the `expert transformer` and advanced captioning.

## 6.2. Ablation Studies / Parameter Analysis
Ablation studies were conducted to validate the effectiveness of key architectural and training components.

### 6.2.1. Position Embedding
`CogVideoX` compared `3D RoPE` with sinusoidal absolute position embedding.
Figure 10 (a) shows the loss curve comparison.

![Figure 10: Training loss curve of different ablations.](images/10.jpg)  
*该图像是多个实验结果的对比图，其中包含四个子图：分别比较RoPE与Sinusoidal、3D与2D+1D注意力机制、不同架构，以及有无显式均匀采样的效果。这些实验旨在展示模型在不同设置下的表现变化。*

Figure 10: Training loss curve of different ablations.

The plot indicates that `RoPE` (green line) converges significantly faster and achieves a lower loss compared to `sinusoidal absolute position embedding` (blue line). This confirms the effectiveness of `RoPE` for modeling long video sequences, consistent with its benefits observed in large language models.

### 6.2.2. Expert Adaptive LayerNorm
The paper compared three architectures: `MMDiT`, `Expert AdaLN (CogVideoX)`, and a model `without Expert AdaLN`.
Figure 8 shows the results, specifically (a), (c) for FVD/Loss and (d) for CLIP4Clip score.

![Figure 8: Ablation studies on WebVid test dataset with 500 videos. MMDiT1 has the same number of parameters with the expert AdaLN. MMDiT2 has the same number of layers but twice number of parameters. a, b, c measure FVD, d measures CLIP4Clip score.](images/8.jpg)  
*该图像是图表，展示了不同实验条件下的训练步骤与性能指标的关系。图中的四个子图分别标记为(a)架构、(b)注意力、(c)显式均匀采样和(d)架构。每个子图中，红色和蓝色的曲线代表不同的实验设置，使用了专家自适应归一化与否；同时，绿色和黄色的标记表示其他对比模型。图中数据反映了在不同训练步骤下的表现变化。*

Figure 8: Ablation studies on WebVid test dataset with 500 videos. MMDiT1 has the same number of parameters with the expert AdaLN. MMDiT2 has the same number of layers but twice number of parameters. a, b, c measure FVD, d measures CLIP4Clip score.

*   `Expert AdaLN` (green lines) significantly outperforms the model `without expert AdaLN` (red lines) and `MMDiT` (blue lines) with the same number of parameters (MMDiT1) in terms of `FVD` (Figure 8a) and `CLIP4Clip` score (Figure 8d). Even `MMDiT2` (yellow lines), which has twice the parameters, is sometimes outperformed or matched by `Expert AdaLN`.
*   The `loss` curve for `Expert AdaLN` (Figure 10c) also shows a more stable and lower trajectory compared to the model without it.
    This suggests that `Expert Adaptive LayerNorm` is highly effective in aligning feature spaces across text and video modalities with minimal additional parameters, making it a more efficient and performant design than using separate transformers or no specific modality alignment mechanism.

### 6.2.3. 3D Full Attention
The effectiveness of `3D full attention` was compared against the common `2D + 1D attention` approach.
Figure 8 (b) shows the FVD comparison, and Figure 10 (b) shows the loss curve.

*   When `3D full attention` (green lines) is used, `FVD` is much lower than `2D + 1D attention` (red lines) in early training steps (Figure 8b), indicating better generation quality.
*   The `loss` curve for `3D full attention` (Figure 10b) is also more stable and lower.
*   The paper observes that `2D + 1D attention` is `unstable` and `prone to collapse`, especially as model size increases. This validates the `3D full attention` approach for its stability and ability to handle complex video dynamics, directly addressing the limitations discussed in Section 2.2 regarding `large motion` consistency.

### 6.2.4. Explicit Uniform Sampling
The impact of `Explicit Uniform Sampling` on training stability and performance was evaluated.
Figure 8 (c) shows the FVD comparison, and Figure 10 (d) shows the loss curve.
The following are the results from Table 9 of the original paper:

<table>
<thead>
<tr>
<td>Timestep</td>
<td>100</td>
<td>300</td>
<td>500</td>
<td>700</td>
<td>900</td>
</tr>
</thead>
<tbody>
<tr>
<td>w/o explicit uniform sampling</td>
<td>0.222</td>
<td>0.130</td>
<td>0.119</td>
<td>0.133</td>
<td>0.161</td>
</tr>
<tr>
<td>w/ explicit uniform sampling</td>
<td>0.216</td>
<td>0.126</td>
<td>0.116</td>
<td>0.129</td>
<td>0.157</td>
</tr>
</tbody>
</table>

Table 9: Validation loss at different diffusion timesteps when the training steps is 40k.

*   `Explicit Uniform Sampling` (green lines) results in a `more stable decrease in loss` (Figure 10d) and achieves a `better FVD` (Figure 8c).
*   Table 9 further shows that `Explicit Uniform Sampling` leads to `lower validation loss` at all diffusion timesteps (e.g., 0.216 vs 0.222 at timestep 100, 0.157 vs 0.161 at timestep 900).
    This suggests that by reducing randomness in timestep sampling, the method effectively stabilizes training and accelerates convergence, leading to improved overall performance.

## 6.3. Inference Time and Memory Consumption
The following are the results from Table 7 of the original paper:

<table>
<thead>
<tr>
<td></td>
<td>5b-480x720-6s</td>
<td>5b-768x1360-5s</td>
<td>2b-480x720-6s</td>
<td>2b-768x1360-5s</td>
</tr>
</thead>
<tbody>
<tr>
<td>Time</td>
<td>113s</td>
<td>500s</td>
<td>49s</td>
<td>220s</td>
</tr>
<tr>
<td>Memory</td>
<td>26GB</td>
<td>76GB</td>
<td>18GB</td>
<td>53GB</td>
</tr>
</tbody>
</table>

Table 7: Inference time and memory consumption of CogVideoX. We evaluate the model on bf, H800 with 50 inference steps.

The inference times and memory consumption are provided for different model sizes and resolutions. As expected, higher resolutions and larger models (5B vs 2B) demand more time and memory. For instance, `CogVideoX-5B` generating $768 \times 1360$ videos for 5 seconds takes 500 seconds and uses 76GB of memory on an H800 GPU for 50 inference steps. This indicates that while highly performant, generation of high-resolution, long videos is still resource-intensive.

The following are the results from Table 8 of the original paper:

<table>
<thead>
<tr>
<td></td>
<td>256*384*6s</td>
<td>480*720*6s</td>
<td>768*1360*5s</td>
</tr>
</thead>
<tbody>
<tr>
<td>2D+1D</td>
<td>0.38s</td>
<td>1.26s</td>
<td>4.17s</td>
</tr>
<tr>
<td>3D</td>
<td>0.41s</td>
<td>2.11s</td>
<td>9.60s</td>
</tr>
</tbody>
</table>

Table 8: Inference time comparison between 3D Full attention and 2D+1D attention. We evaluate the model on bf, H800 with one dit forward step. Thanks to the optimization by Flash Attention, the increase in sequence length does not make the inference time unacceptable.

Comparing `3D Full attention` with `2D+1D attention` for a single `DiT` forward step, `3D Full attention` (0.41s, 2.11s, 9.60s) is computationally more expensive than `2D+1D attention` (0.38s, 1.26s, 4.17s) especially at higher resolutions. However, the authors note that `Flash Attention` optimization makes this increase in inference time acceptable, especially given the significant quality and stability benefits of `3D Full attention`.

## 6.4. Data Filtering Details
The following are the results from Table 14 of the original paper:

<table>
<thead>
<tr>
<td>Classifier</td>
<td>TP</td>
<td>FP</td>
<td>TN</td>
<td>FN</td>
<td>Test Acc</td>
</tr>
</thead>
<tbody>
<tr>
<td>Classifier - Editing</td>
<td>0.81</td>
<td>0.02</td>
<td>0.09</td>
<td>0.08</td>
<td>0.91</td>
</tr>
<tr>
<td>Classifier - Static</td>
<td>0.48</td>
<td>0.04</td>
<td>0.44</td>
<td>0.04</td>
<td>0.92</td>
</tr>
<tr>
<td>Classifier - Lecture</td>
<td>0.52</td>
<td>0.00</td>
<td>0.47</td>
<td>0.01</td>
<td>0.99</td>
</tr>
<tr>
<td>Classifier - Text</td>
<td>0.60</td>
<td>0.03</td>
<td>0.36</td>
<td>0.02</td>
<td>0.96</td>
</tr>
<tr>
<td>Classifier - Screenshot</td>
<td>0.61</td>
<td>0.01</td>
<td>0.37</td>
<td>0.01</td>
<td>0.98</td>
</tr>
<tr>
<td>Classifier - Low Quality</td>
<td>0.80</td>
<td>0.02</td>
<td>0.09</td>
<td>0.09</td>
<td>0.89</td>
</tr>
</tbody>
</table>

Table 14: Summary of Classifiers Performance on the Test Set. TP: True Positive, FP: False Positive, TN: True Negative, FN: False Negative.

The performance of the `Video-LLaMA`-based classifiers for filtering low-quality data is strong. The classifiers achieve high test accuracies ranging from 0.89 (`Low Quality`) to 0.99 (`Lecture`). This indicates that the data filtering pipeline is effective at identifying and removing various types of undesirable video content, thereby ensuring that the model is trained on a high-quality dataset, which is crucial for the observed generation quality.

Overall, the results consistently support the efficacy of `CogVideoX`'s novel designs and training strategies in pushing the boundaries of text-to-video generation, particularly in terms of motion, temporal coherence, and semantic alignment.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper successfully introduces `CogVideoX`, a state-of-the-art text-to-video diffusion model designed to overcome critical limitations in generating long, coherent, and dynamic videos from text prompts. The key innovations include:
*   A `3D Causal VAE` for highly efficient and high-fidelity spatial-temporal video compression, which significantly reduces flickering and improves reconstruction quality.
*   An `Expert Transformer` equipped with `Expert Adaptive LayerNorm` to facilitate deep and efficient fusion between text and video modalities, enhancing semantic alignment and reducing parameter overhead.
*   The adoption of `3D Full Attention` to comprehensively model video data across both spatial and temporal dimensions, crucial for capturing large-scale motions and ensuring temporal consistency, supported by optimizations like `FlashAttention`.
*   Robust training strategies, including `progressive training` across resolutions and `Multi-Resolution Frame Pack`, to effectively utilize diverse data and improve generalization.
*   `Explicit Uniform Sampling`, a novel timestep sampling method that stabilizes training loss and accelerates convergence.
*   A sophisticated `Dense Video Caption Data Generation pipeline` that provides rich, high-quality textual descriptions for training data, significantly improving the model's understanding of video content.

    Evaluations, both automated (e.g., `Dynamic Degree`, `GPT4o-MTScore`, `FVD`) and human-based (against a strong closed-source model `Kling`), consistently demonstrate `CogVideoX-5B`'s superior performance, setting new benchmarks for text-to-video generation quality, dynamism, and instruction following. The public release of `CogVideoX` models and associated components further contributes to advancing the field.

## 7.2. Limitations & Future Work
The authors implicitly and explicitly acknowledge several areas for improvement and future research:
*   **Larger Compression Ratios for 3D VAE:** The paper notes that exploring VAEs with larger compression ratios is a future work. While their current `3D VAE` is effective, further compression without sacrificing quality would be beneficial for even longer or higher-resolution videos. The challenge of `model convergence` with aggressive spatial-temporal compression (e.g., $16 \times 16 \times 8$) is noted.
*   **Scaling Laws:** The paper mentions they are "exploring the scaling laws of video generation models and aim to train larger and more powerful models." This implies that while `CogVideoX-5B` is state-of-the-art, further scaling of model parameters, data volume, and training compute could lead to even better performance, suggesting current models might not be at the peak of their potential.
*   **Semantic Degradation from Fine-Tuning:** In the `High-Quality Fine-Tuning` stage, while visual quality improved and artifacts were removed, the authors observed "a slight degradation in the model's semantic ability." This indicates a trade-off that needs to be addressed, potentially by refining the fine-tuning process or the data selection for this stage.
*   **Computational Cost of High-Resolution, Long-Duration Generation:** While `FlashAttention` helps, generating $768 \times 1360$ videos for 5 seconds still takes 500 seconds and 76GB of memory on an H800, highlighting the resource intensity. Future work could focus on further inference optimizations or more efficient architectures.

## 7.3. Personal Insights & Critique
`CogVideoX` represents a significant step forward in text-to-video generation, particularly in its rigorous approach to temporal consistency and motion.
*   **Innovation in Modality Fusion:** The `Expert Adaptive LayerNorm` is a clever and efficient solution to the perennial problem of integrating disparate modalities (text and vision). It provides a mechanism for deep fusion within a unified transformer architecture, which is more parsimonious than separate encoders and likely contributes to the model's strong `Instruction Following` scores. This concept could be transferable to other multi-modal tasks where heterogeneous data needs to be integrated efficiently.
*   **Emphasis on Data Quality and Curation:** The extensive `data filtering pipeline` and the sophisticated `Dense Video Caption Data Generation pipeline` highlight the critical role of high-quality data in achieving state-of-the-art results. This reinforces the idea that model architecture alone is not enough; carefully curated and richly annotated data is paramount. The `GPT-4` driven captioning is a powerful approach for generating granular, descriptive labels, a bottleneck for many video tasks.
*   **Addressing Fundamental Video Challenges:** The dedication to `3D Causal VAE` and `3D Full Attention` directly tackles the core difficulties of video: `temporal redundancy` and `motion coherence`. Many previous models have "cheated" by relying on 2D architectures or separated attention, leading to artifacts. `CogVideoX`'s direct approach, enabled by `FlashAttention` and progressive training, is more principled.
*   **Impact of Open-Sourcing:** The public release of `CogVideoX` models, including `VAE` and `video captioning`, is a monumental contribution. It democratizes access to powerful video generation capabilities, enabling wider research, application development, and potentially fostering a vibrant ecosystem around these models. This is crucial for accelerating progress in the field, much like `Stable Diffusion` did for image generation.
*   **Potential Unverified Assumptions/Areas for Improvement:**
    *   **"Commercial-grade" claim:** While impressive, the term "commercial-grade" needs further scrutiny regarding robustness, ethical considerations, and deployability in diverse real-world scenarios. The human evaluation framework is a good start but specific failure modes (e.g., hallucination, biases, safety) are not extensively discussed.
    *   **Generalizability of 3D-RoPE extrapolation:** While chosen for clarity, `extrapolation` in positional embeddings can sometimes lead to less robust generalization to vastly different resolutions or aspect ratios not seen during training. Further analysis on its behavior at extreme aspect ratios might be beneficial.
    *   **Trade-off in Fine-tuning:** The observed `slight degradation in semantic ability` during high-quality fine-tuning points to a subtle challenge. It might imply that aesthetic filtering can sometimes remove data points that, while visually imperfect, contain rich semantic information valuable for the model's understanding. Investigating ways to maintain semantic richness while enhancing visual quality would be valuable.
    *   **Energy Consumption:** High inference times and memory footprints, especially for `CogVideoX-5B`, imply substantial energy consumption. As models scale, energy efficiency will become an increasingly important consideration for environmental impact and practical deployment.