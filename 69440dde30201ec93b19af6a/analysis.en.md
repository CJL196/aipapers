# 1. Bibliographic Information

## 1.1. Title
HunyuanVideo: A Systematic Framework For Large Video Generative Models

## 1.2. Authors
The paper is authored by the **Hunyuan Foundation Model Team** from Tencent. This indicates a large-scale, industry-led research and engineering effort rather than a small academic group. Tencent is a major technology conglomerate with significant resources in AI research and large-scale computing infrastructure.

## 1.3. Journal/Conference
The paper was published on **arXiv**, an open-access repository for electronic preprints of scientific papers. As a preprint, it has not yet undergone a formal peer-review process for an academic conference or journal. arXiv is a highly reputable and influential platform in fields like machine learning, allowing for the rapid dissemination of cutting-edge research to the global community.

## 1.4. Publication Year
2024. The paper was submitted on December 3, 2024.

## 1.5. Abstract
The abstract introduces `HunyuanVideo`, an open-source video foundation model designed to close the performance gap between proprietary, closed-source models (like Sora) and publicly available ones. The authors present a comprehensive framework covering four key areas: data curation, advanced model architecture, progressive scaling and training strategies, and an efficient infrastructure for large-scale operations. This framework enabled the training of a 13 billion parameter video generative model, the largest in the open-source domain. The paper highlights that targeted designs were implemented to ensure high visual quality, realistic motion, strong text-video alignment, and advanced cinematic effects. Professional human evaluations show that `HunyuanVideo` outperforms leading models, including Runway Gen-3 and Luma 1.6. By open-sourcing the model and its application code, the team aims to empower the research community and foster a more vibrant video generation ecosystem.

## 1.6. Original Source Link
- **Original Source Link:** [https://arxiv.org/abs/2412.03603](https://arxiv.org/abs/2412.03603)
- **PDF Link:** [https://arxiv.org/pdf/2412.03603v6.pdf](https://arxiv.org/pdf/2412.03603v6.pdf)
- **Publication Status:** This is a preprint paper available on arXiv and has not been peer-reviewed.

# 2. Executive Summary

## 2.1. Background & Motivation
The field of video generation has seen remarkable progress, but the most powerful and high-performing models (e.g., OpenAI's Sora, Google's MovieGen) remain **closed-source**. This creates a significant **performance gap** between what industry leaders can achieve and what is accessible to the broader academic and open-source communities. This disparity stifles innovation, as researchers and developers lack access to strong foundational models to build upon, experiment with, and improve. The paper argues that the relative stagnation in open-source video generation, compared to the thriving open-source image generation ecosystem, is largely due to this lack of a powerful, publicly available base model.

The innovative idea of `HunyuanVideo` is to tackle this problem head-on by not just releasing a model, but by building and open-sourcing a **systematic framework** for creating large-scale video models. This includes the entire pipeline from data processing to model training and deployment, providing a blueprint for the community.

## 2.2. Main Contributions / Findings
The paper's main contributions are:

1.  **A Comprehensive Open-Source Framework:** The authors introduce `HunyuanVideo`, a complete system for large-scale video generation. It details solutions for four critical components:
    *   **Data Curation:** A hierarchical filtering and structured recaptioning pipeline to create high-quality, diverse training data.
    *   **Model Architecture:** A unified image-video Transformer architecture with a custom 3D VAE and an advanced Multimodal Large Language Model (MLLM) as a text encoder.
    *   **Efficient Training & Scaling:** The paper derives empirical scaling laws for video models to optimize the trade-off between model size, data size, and computation, leading to a more efficient training process.
    *   **Scalable Infrastructure:** Details on the parallelization strategies and optimizations used to train a massive model efficiently.

2.  **State-of-the-Art 13B Open-Source Model:** The team successfully trained and released a 13 billion parameter video generation model, which was the largest open-source model of its kind at the time of publication.

3.  **Superior Performance:** Through extensive human evaluation involving 60 professionals and over 1,500 prompts, `HunyuanVideo` was shown to outperform leading closed-source competitors like Runway Gen-3 and Luma 1.6, particularly in generating high-quality motion.

4.  **Versatile Downstream Applications:** The paper demonstrates the power of the foundation model by showcasing its adaptability to various downstream tasks, including video-to-audio generation, image-to-video conversion, and fully controllable avatar animation (driven by audio, pose, and expressions).

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that learn to create data by reversing a process of gradually adding noise. The process has two parts:
*   **Forward Process:** This is a fixed process where you start with a real data sample (e.g., an image or video) and repeatedly add a small amount of Gaussian noise over many steps. After enough steps, the original data is transformed into pure, unstructured noise.
*   **Reverse Process (Denoising):** The model, typically a neural network like a U-Net or a Transformer, is trained to reverse this process. At each step, it takes a noisy sample and predicts the noise that was added. By subtracting this predicted noise, it can gradually denoise the sample, step-by-step, until a clean data sample is generated from pure noise. This denoising process is conditioned on extra information, like a text prompt, to guide the generation.

### 3.1.2. Flow Matching
Flow Matching is a more recent and often more efficient alternative to diffusion models. Instead of learning to reverse a stochastic (random) noising process, Flow Matching learns to model a **deterministic path** (a continuous flow) from a simple probability distribution (like Gaussian noise) to the complex distribution of the real data. The model is trained to predict the **velocity vector** at any point along this path. During inference, it starts with a random noise sample and integrates these velocities over time using an Ordinary Differential Equation (ODE) solver to deterministically transform the noise into a coherent data sample. This can lead to faster and more stable training and generation.

### 3.1.3. Variational Autoencoder (VAE)
A Variational Autoencoder (VAE) is a generative model used for data compression and generation. It consists of two main parts:
*   **Encoder:** A neural network that takes a high-dimensional input (like an image) and compresses it into a low-dimensional representation in a **latent space**. This latent space is a probabilistic distribution, typically a Gaussian.
*   **Decoder:** Another neural network that takes a point sampled from the latent space and reconstructs the original high-dimensional data.
    In the context of large generative models like Stable Diffusion and `HunyuanVideo`, VAEs are used to create a **latent diffusion model**. The computationally expensive diffusion/flow matching process happens in the small, compressed latent space, not the massive pixel space. Once a latent representation is generated, the lightweight VAE decoder converts it back into a full-resolution image or video. `HunyuanVideo` uses a **3D VAE** to handle the temporal dimension of video.

### 3.1.4. Transformer and Self-Attention
The Transformer is a neural network architecture that has become dominant in natural language processing (NLP) and is now widely used in computer vision. Its core innovation is the **self-attention mechanism**.
*   **Self-Attention:** This mechanism allows the model to weigh the importance of different parts of the input sequence when processing a specific part. For an image or video, this means every pixel (or patch) can "look at" every other pixel to understand the global context and relationships within the scene. In `HunyuanVideo`, a `Full Attention` mechanism is used, meaning every spatio-temporal token can attend to every other token, enabling the model to learn complex relationships across both space and time.

    The attention score is calculated using the following formula:
\$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\$
*   **Symbol Explanation:**
    *   $Q$ (Query): A matrix representing the current token/patch being processed.
    *   $K$ (Key): A matrix representing all tokens/patches in the sequence, used for comparison with the Query.
    *   $V$ (Value): A matrix representing the content of all tokens/patches.
    *   $d_k$: The dimension of the Key vectors. The division by $\sqrt{d_k}$ is a scaling factor to stabilize gradients.
    *   `softmax`: A function that converts the raw scores into a probability distribution, ensuring the weights sum to 1. The output is a weighted sum of the Value vectors, where the weights are determined by the similarity between the Query and Keys.

### 3.1.5. Classifier-Free Guidance (CFG)
Classifier-Free Guidance is a technique to improve how well a conditional diffusion model follows its guidance (e.g., a text prompt) without needing a separate classifier model. During training, the model is sometimes trained with the text condition and sometimes without it (unconditionally). During inference, the model makes two predictions at each step: one conditional ($\epsilon_{\theta}(x_t, c)$) and one unconditional ($\epsilon_{\theta}(x_t)$). The final prediction is an extrapolation away from the unconditional prediction and towards the conditional one, controlled by a guidance scale $w$:
`final_prediction` = $\epsilon_{\theta}(x_t) + w \cdot (\epsilon_{\theta}(x_t, c) - \epsilon_{\theta}(x_t))$.
A higher $w$ forces the model to adhere more strictly to the prompt, often at the cost of sample diversity. `HunyuanVideo` uses a distillation technique to avoid the computational cost of making two predictions at inference time.

## 3.2. Previous Works
The paper positions itself relative to several key models in the video generation landscape:
*   **Closed-Source SOTA:** Models like **OpenAI's Sora** [7] and **Google's MovieGen** [67] are the benchmarks for high-fidelity, long-duration, and physically plausible video generation. They are noted for their "world simulator" capabilities but are inaccessible to the public, creating the motivation for `HunyuanVideo`.
*   **Open-Source Efforts:**
    *   **Stable Video Diffusion (SVD)** [5]: A latent diffusion model based on an image model, extended for video. It's a strong open-source baseline but limited in duration and resolution.
    *   **Open-Sora** [102]: A community effort to replicate the architecture of Sora, also using a Diffusion Transformer (`DiT`) backbone. `HunyuanVideo` differentiates itself with its massive scale and complete, systematic framework.
    *   **FLUX** [47]: An image generation model that introduced the "dual-stream to single-stream" architecture to better integrate text and image information. `HunyuanVideo` adapts this concept for video.
    *   **DiT (Diffusion Transformer)** [65]: This work replaced the commonly used U-Net backbone in diffusion models with a Transformer, showing excellent scalability and performance, which heavily influenced the architectural choices for Sora and `HunyuanVideo`.

## 3.3. Technological Evolution
The technology for generative modeling has evolved rapidly:
1.  **GANs (Generative Adversarial Networks):** Early leaders in image synthesis, but often difficult to train and control.
2.  **Pixel-Space Diffusion Models:** Models like DDPM demonstrated high-quality generation but were computationally very slow as they operated on the full pixel grid.
3.  **Latent Diffusion Models (LDMs):** Models like Stable Diffusion introduced the use of a VAE to compress images into a smaller latent space, making the diffusion process much more efficient. This was a major breakthrough.
4.  **Diffusion Transformers (DiTs):** The latest evolution involves replacing the U-Net backbone in LDMs with a Transformer. Transformers have proven to scale better with more data and parameters, leading to the state-of-the-art performance seen in models like Sora and `HunyuanVideo`.

    `HunyuanVideo` stands at the current frontier, representing a large-scale, open-source implementation of the Diffusion Transformer paradigm for video, enhanced with a complete ecosystem for data, training, and deployment.

## 3.4. Differentiation Analysis
Compared to previous work, `HunyuanVideo`'s core innovations are:
*   **Systematic and Holistic Approach:** It's not just a model but a complete, documented framework. The paper's emphasis on data curation, scaling laws, and infrastructure is a unique and valuable contribution.
*   **Unprecedented Open-Source Scale:** At 13 billion parameters, it significantly surpasses previous open-source models in size, allowing it to capture more complex dynamics and details.
*   **Custom Components:** Instead of relying entirely on off-the-shelf parts, the team trained a high-performance **3D VAE from scratch** and utilized a powerful **MLLM as a text encoder**, providing finer control and better performance.
*   **Data-Driven Rigor:** The derivation of **empirical scaling laws** for video generation provided a principled way to allocate computational budget, a practice common in LLM training but less documented for video models.
*   **Structured Captioning:** The use of an in-house VLM to generate detailed, multi-faceted JSON captions goes far beyond the simple descriptive captions used in many datasets, enabling better control and alignment.

# 4. Methodology
`HunyuanVideo` is presented as a comprehensive system. Its methodology can be broken down into four main pillars: data pre-processing, model architecture design, model training strategy, and model acceleration.

The overall training system is depicted in the figure below.

![Figure 3: The overall training system for Hunyuan Video.](images/2.jpg)
*该图像是Hunyuan Video的整体训练系统示意图，展示了数据预处理、模型训练和应用三个主要部分。首先，通过数据过滤和结构化标注处理图像和视频数据，然后进入多阶段训练的大规模模型训练，最后实现图像到视频生成、上半身和全身虚拟形象生成等多种应用。*

## 4.1. Data Pre-processing
High-quality data is the foundation of the model. The team employs a meticulous, multi-stage process for data curation and annotation.

### 4.1.1. Hierarchical Data Filtering
Starting from a large raw data pool of images and videos, a hierarchical filtering pipeline is used to create progressively higher-quality datasets for different training stages.

The pipeline, shown in Figure 4, involves several steps:
1.  **Shot Segmentation:** Raw videos are split into single-shot clips using `PySceneDetect`.
2.  **Deduplication & Balancing:** A `VideoCLIP` model is used to generate embeddings for each clip. These embeddings are used to remove duplicate clips and to perform k-means clustering to balance the concepts in the dataset.
3.  **Multi-faceted Filtering:** A series of specialized models are used to filter clips based on various quality criteria:
    *   **Aesthetics & Technical Quality:** `Dover` is used to assess visual appeal.
    *   **Clarity:** A custom model removes blurry clips.
    *   **Motion:** Optical flow is used to filter out static or very slow videos.
    *   **Content:** OCR models remove clips with excessive text, and object detection models remove watermarks and borders.
4.  **Progressive Datasets:** By progressively increasing the strictness of these filters, multiple datasets are created, with resolutions increasing from 256p to 720p. A final high-quality fine-tuning dataset of ~1 million samples is curated manually by human annotators who rate clips on detailed aesthetic and motion criteria.

    ![Figure 4: Our hierarchical data filtering pipeline. We employ various filters for data filtering and progressively increase their thresholds to build 4 training datasets, i.e., 256p, 360p, 540p, and $7 2 0 \\mathrm { p }$ , while the final SFT dataset is built through manual annotation. This figure highlights some of the most important filters to use at each stage. A large portion of data will be removed at each stage, ranging from half to one-fifth of the data from the previous stage. Here, gray bars represent the amount of data filtered out by each filter while colored bars indicate the amount of remaining data at each stage.](images/3.jpg)
    *该图像是一个示意图，展示了层级数据过滤管道。图中使用了多种过滤器，通过逐步增加阈值构建了256p、360p、540p和720p四个训练数据集，最终的SFT数据集通过人工标注构建。各个阶段过滤器移除了大量数据，灰色条表示被过滤的数据量，彩色条表示剩余的数据量。*

### 4.1.2. Structured Data Annotation
To improve the model's ability to understand and follow complex prompts, the team developed an in-house Vision Language Model (VLM) to generate structured captions in JSON format for all data. These captions are multi-dimensional, including fields for:
*   `Short Description`: Main content summary.
*   `Dense Description`: Detailed scene content, including camera movements.
*   `Background`: Environmental context.
*   `Style`: Type of shot (e.g., `aerial shot`, `close-up`).
*   `Lighting` and `Atmosphere`.
    A separate classifier also predicts 14 types of camera movements (e.g., `pan left`, `zoom in`), which are added to the captions to enable explicit camera control.

## 4.2. Core Methodology In-depth: Model Architecture
The core of `HunyuanVideo` is a Transformer-based generative model operating in a latent space.

The overall architecture is shown below.

![Figure 5: The overall architecture of HunyuanVideo. The model is trained on a spatial-temporally compressed latent space, which is compressed through Causal 3D VAE. Text prompts are encoded using a large language model, and used as the condition. Gaussian noise and condition are taken as input, our model generates a output latent, which is decoded into images or videos through the 3D VAE decoder.](images/4.jpg)
*该图像是HunyuanVideo系统的整体架构示意图。模型在空间时间压缩的潜在空间中训练，利用Causal 3D VAE进行压缩。文本提示使用大型语言模型编码作为条件输入，加上高斯噪声，生成的潜在输出通过3D VAE解码为图像或视频。*

### 4.2.1. Causal 3D Variational Autoencoder (3D VAE)
The first component is a 3D VAE that compresses videos into a compact latent representation. This reduces the computational load for the main generative model.
*   **Architecture:** It uses `CausalConv3D` layers, which are 3D convolutions that only look at past frames, making them suitable for video processing. It compresses a video by a factor of 4 in the temporal dimension ($c_t=4$) and 8 in each spatial dimension ($c_s=8$). The latent representation has 16 channels ($C=16$).
*   **Training:** Unlike many other models that initialize from a pre-trained image VAE, the `HunyuanVideo` VAE is **trained from scratch** on a mix of video and image data (4:1 ratio). This allows it to be better optimized for video reconstruction. The training objective is a combination of four losses:
    \$
    \mathrm { Loss } = L _ { 1 } + 0 . 1 L _ { l p i p s } + 0 . 0 5 L _ { a d v } + 1 0 ^ { - 6 } L _ { k l }
    \$
    *   **Symbol Explanation:**
        *   $L_1$: The L1 reconstruction loss, measuring the absolute difference between the original and reconstructed pixels.
        *   $L_{lpips}$: The Perceptual Loss (Learned Perceptual Image Patch Similarity), which uses features from a deep neural network to measure perceptual similarity, leading to visually more pleasing reconstructions.
        *   $L_{adv}$: An adversarial GAN loss, where a discriminator tries to distinguish between real and reconstructed videos, pushing the VAE to produce more realistic outputs.
        *   $L_{kl}$: The Kullback-Leibler divergence loss, a regularization term standard in VAEs that ensures the latent space is well-structured (close to a standard normal distribution).
*   **Inference:** For high-resolution videos, a **spatial-temporal tiling strategy** is used. The video is split into overlapping tiles, each is decoded separately, and the results are blended together.

    The architecture of the 3D VAE is as follows:

    ![Figure 6: The architecture of our 3DVAE.](images/5.jpg)
    *该图像是一个示意图，展示了3DVAE架构的编码器和解码器部分。左侧为输入的多维数据，经过CausalConv3D编码器处理后，输出的特征图形状为 $\left( \frac{T}{4}+1, \frac{H}{8}, \frac{W}{8} \right)$，再通过CausalConv3D解码器生成最终输出。*

### 4.2.2. Unified Image and Video Generative Architecture
The main generative model is a **Diffusion Transformer (`DiT`)** that operates on the latent space provided by the 3D VAE.

![Figure 8: The architecture of our HunyuanVideo Diffusion Backbone.](images/7.jpg)
*该图像是HunyuanVideo扩散骨干网的架构示意图。它展示了模型中包括双流和单流的DiT块，以及数据处理和特征提取的流程，包括从输入的噪声到输出的生成结果的各个步骤。*

*   **Backbone:** The architecture uses `Full Attention` Transformer blocks, where every spatio-temporal token can attend to every other token. This unified design simplifies the model and allows it to process both images (as single-frame videos) and videos.
*   **"Dual-stream to Single-stream" Design:**
    1.  **Dual-stream Phase:** In the initial layers, video tokens and text tokens are processed in separate Transformer blocks. This allows each modality to learn its own representations without interference.
    2.  **Single-stream Phase:** In the later layers, the video and text tokens are concatenated and fed into a single set of Transformer blocks. This enables deep fusion and complex cross-modal interactions.
*   **Position Embedding:** To handle variable resolutions, aspect ratios, and durations, the model uses **3D Rotary Position Embedding (RoPE)**. The feature channels of the query and key vectors in the attention mechanism are split into three parts, and each part is rotated based on its coordinate in the time (T), height (H), and width (W) dimensions, respectively. This allows the model to understand both absolute and relative positions in the 3D spatio-temporal grid.
*   **Text Encoder:** Instead of standard text encoders like T5, `HunyuanVideo` uses a pre-trained **Multimodal Large Language Model (MLLM)**. The authors argue MLLMs have better image-text alignment and complex reasoning abilities after instruction tuning. Since MLLMs are decoder-only (causal attention), a **bidirectional token refiner** is added to produce better text guidance features. `CLIP` text features are also used as a global summary of the prompt.

    ![Figure 9: Text encoder comparison between T5 XXL and the instruction-guided MLLM introduced by HunyuanVideo.](images/8.jpg)
    *该图像是图表，展示了T5 XXL和HunyuanVideo引入的指令引导的多模态大语言模型（MLLM）之间的文本编码器比较。图中标注了双向注意力和因果注意力的不同，体现了各自的架构设计。*

## 4.3. Model Scaling and Training
A key contribution is the systematic approach to scaling and training the 13B parameter model.

### 4.3.1. Scaling Laws
To efficiently allocate their computational budget, the team derived empirical scaling laws, similar to those for LLMs. They model the relationship between computation ($C$), optimal model size ($N_{opt}$), and optimal dataset size ($D_{opt}$).
\$
N _ { o p t } = a _ { 1 } C ^ { b _ { 1 } } , \quad D _ { o p t } = a _ { 2 } C ^ { b _ { 2 } }
\$
*   **Image Model Scaling:** They first trained a family of `DiT` models for text-to-image generation (`DiT-T2X(I)`) from 92M to 6.6B parameters and fitted the scaling law coefficients.
*   **Video Model Scaling:** Using the optimal image models as initialization, they then trained video models (`DiT-T2X(V)`) and fitted a new set of scaling law coefficients.
    Based on these laws, they determined that a **13B parameter model** was the optimal choice for their available resources. The scaling laws are visualized in the figure below.

    ![该图像是六个子图的组合，其中展示了 T2X(I) 的损失曲线、幂律关系及相关参数的变化。子图 (a) 表示损失曲线和封闭点，(b) 至 (c) 展示了 C 和 N、C 及 D 的幂律关系，(d) 至 (f) 分别为 T2X(V) 的损失曲线和相关参数变化。公式中涉及的幂律关系可表示为 $y heta x^k$。](images/9.jpg)
    *该图像是六个子图的组合，其中展示了 T2X(I) 的损失曲线、幂律关系及相关参数的变化。子图 (a) 表示损失曲线和封闭点，(b) 至 (c) 展示了 C 和 N、C 及 D 的幂律关系，(d) 至 (f) 分别为 T2X(V) 的损失曲线和相关参数变化。公式中涉及的幂律关系可表示为 $y heta x^k$。*

### 4.3.2. Training Objective and Strategy
*   **Training Objective:** The model is trained using the **Flow Matching** framework. The objective is to minimize the Mean Squared Error between the model's predicted velocity field $\mathbf{v}_t$ and the ground truth velocity $\mathbf{u}_t$ that transforms noise into data.
    \$
    \mathcal { L } _ { \mathrm { g e n e r a t i o n } } = \mathbb { E } _ { t , { \mathbf { x } _ { 0 } } , { \mathbf { x } _ { 1 } } } \| \mathbf { v } _ { t } - { \mathbf { u } } _ { t } \| ^ { 2 }
    \$
    *   **Symbol Explanation:**
        *   $\mathbf{x}_1$: A real video latent from the training data.
        *   $\mathbf{x}_0$: A sample of random Gaussian noise.
        *   $t$: A time step sampled from `[0, 1]`.
        *   $\mathbf{u}_t$: The ground truth velocity of the straight path from $\mathbf{x}_0$ to $\mathbf{x}_1$ at time $t$.
        *   $\mathbf{v}_t$: The velocity predicted by the model for a sample $\mathbf{x}_t$ at time $t$.

*   **Progressive Training:** A curriculum learning strategy is employed to stabilize training and improve quality:
    1.  **Image Pre-training:** The model is first pre-trained on images only, first at 256px and then using a mix-scale strategy (256px and 512px) to learn high-resolution details without forgetting low-resolution concepts.
    2.  **Video-Image Joint Training:** The model, initialized with image pre-trained weights, is then trained jointly on images and videos. The training progresses through stages of increasing resolution and duration: low-res short videos -> low-res long videos -> high-res long videos.
    3.  **High-Performance Fine-tuning:** Finally, the model is fine-tuned on a small, manually curated dataset of extremely high-quality videos to maximize visual appeal and motion dynamics.

## 4.4. Model Acceleration
To make training and inference practical for a 13B model, several acceleration techniques were used.

### 4.4.1. Inference Acceleration
*   **Inference Step Reduction:** To generate videos with fewer steps (e.g., 10 instead of 50), they use a **time-step shifting** strategy. Instead of sampling time steps linearly, they use a function that concentrates more steps at the beginning of the generation process, where most of the structure is formed.
*   **Text-Guidance Distillation:** To eliminate the 2x computational cost of Classifier-Free Guidance (CFG), they distill the behavior of the guided model into a single student model. This student model takes the guidance scale as an input and is trained to directly output the guided prediction, resulting in a ~1.9x speedup.

### 4.4.2. Training Infrastructure and Parallelism
Training was performed on Tencent's `AngelPTM` framework.
*   **5D Parallelism:** A combination of five parallelization strategies was used to distribute the model and data across GPUs:
    *   `Tensor Parallelism (TP)`: Splits model weights within layers.
    *   `Sequence Parallelism (SP)`: Splits the input sequence across GPUs to reduce activation memory.
    *   `Context Parallelism (CP)`: Splits long sequences for attention calculation (using `Ring Attention`).
    *   `Data Parallelism (DP)`: Processes different data batches on different GPUs.
    *   `ZeroCache`: An optimization similar to ZeRO, which partitions model states (parameters, gradients, optimizer states) to reduce memory redundancy.
*   **Other Optimizations:** `FusedAttention` for faster attention computation and `Recomputation` (gradient checkpointing) to trade compute for memory were also employed.

# 5. Experimental Setup

## 5.1. Datasets
*   **Training Data:** The authors used a massive, proprietary, internal data pool of billions of image-text pairs and a large collection of videos. The paper does not use standard public benchmarks for training but provides extensive detail on its **data curation process** (see Section 4.1). This process yields several datasets for progressive training and a final fine-tuning dataset of approximately 1 million high-quality, human-annotated video clips. An example of a data sample can be inferred from the prompts used for generation, such as in Figure 12: `A white cat sits on a white soft sofa like a person, while its long-haired male owner, with his hair tied up in a topknot, sits on the floor, gazing into the cat's eyes. His child stands nearby, observing the interaction between the cat and the man.`
*   **Evaluation Data:** The main comparison was conducted on a set of **1,533 representative text prompts**. These prompts were used to generate videos from all models for human evaluation.

## 5.2. Evaluation Metrics
The primary evaluation method was subjective human assessment, supplemented by objective metrics for specific components like the VAE.

### 5.2.1. Human Evaluation Metrics
For the main model comparison, 60 professional evaluators rated the generated videos on a "win rate" basis (which model is better) across three criteria:
*   **Text Alignment:**
    *   **Conceptual Definition:** This metric assesses how accurately the generated video reflects the content, subjects, actions, and relationships described in the text prompt. A high score means the model correctly interpreted and rendered all key elements of the prompt.
*   **Motion Quality:**
    *   **Conceptual Definition:** This measures the realism, fluidity, and coherence of movement in the video. It evaluates whether the motion is physically plausible, dynamic, and free from artifacts like stuttering or unnatural deformations.
*   **Visual Quality:**
    *   **Conceptual Definition:** This evaluates the aesthetic appeal and technical fidelity of the video frames. It considers factors like image clarity, color harmony, lighting, texture detail, and the absence of visual artifacts like blurriness or distortion.

### 5.2.2. Objective Metrics
*   **Peak Signal-to-Noise Ratio (PSNR):**
    *   **Conceptual Definition:** PSNR is used to measure the reconstruction quality of the VAE. It quantifies the ratio between the maximum possible power of a signal (the maximum pixel value) and the power of the corrupting noise that affects the fidelity of its representation. Higher PSNR generally indicates better reconstruction quality.
    *   **Mathematical Formula:**
        \$
        \mathrm{PSNR} = 10 \cdot \log_{10}\left(\frac{\mathrm{MAX}_I^2}{\mathrm{MSE}}\right)
        \$
    *   **Symbol Explanation:**
        *   $\mathrm{MAX}_I$: The maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image).
        *   $\mathrm{MSE}$: The Mean Squared Error between the original image/video and the reconstructed one.
            \$
        \mathrm{MSE} = \frac{1}{m \cdot n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i, j) - K(i, j)]^2
        \$
        where $I$ is the original image and $K$ is the reconstructed image of size $m \times n$.

## 5.3. Baselines
`HunyuanVideo` was compared against a strong set of contemporary closed-source video generation models:
*   **Runway Gen-3 alpha:** A leading commercial text-to-video model from Runway ML.
*   **Luma 1.6:** A high-quality video generation model from Luma AI, available via API.
*   **CNTopA, CNTopB, CNTopC:** Three top-performing but unnamed commercial video generation models in China.

    These baselines are representative as they were considered state-of-the-art commercial offerings at the time of the study, making the comparison a direct challenge to the best closed-source systems.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The primary results come from the large-scale human evaluation comparing `HunyuanVideo` against five strong baseline models.

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Model Name</th>
<th>Duration</th>
<th>Text Alignment</th>
<th>Motion Quality</th>
<th>Visual Quality</th>
<th>Overall</th>
<th>Ranking</th>
</tr>
</thead>
<tbody>
<tr>
<td>HunyuanVideo (Ours)</td>
<td>5s</td>
<td>61.8%</td>
<td>66.5%</td>
<td>95.7%</td>
<td>41.3%</td>
<td>1</td>
</tr>
<tr>
<td>CNTopA (API)</td>
<td>5s</td>
<td>62.6%</td>
<td>61.7%</td>
<td>95.6%</td>
<td>37.7%</td>
<td>2</td>
</tr>
<tr>
<td>CNTopB (Web)</td>
<td>5s</td>
<td>60.1%</td>
<td>62.9%</td>
<td>97.7%</td>
<td>37.5%</td>
<td>3</td>
</tr>
<tr>
<td>GEN-3 alpha (Web)</td>
<td>6s</td>
<td>47.7%</td>
<td>54.7%</td>
<td>97.5%</td>
<td>27.4%</td>
<td>4</td>
</tr>
<tr>
<td>Luma1.6 (API)</td>
<td>5s</td>
<td>57.6%</td>
<td>44.2%</td>
<td>94.1%</td>
<td>24.8%</td>
<td>5</td>
</tr>
<tr>
<td>CNTopC (Web)</td>
<td>5s</td>
<td>48.4%</td>
<td>47.2%</td>
<td>96.3%</td>
<td>24.6%</td>
<td>6</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Overall Performance:** `HunyuanVideo` achieves the highest **overall win rate of 41.3%**, ranking it #1 among all tested models. This is a significant achievement for an open-source model competing against leading commercial products.
*   **Motion Quality:** The model's standout feature is its **motion quality**, where it wins 66.5% of the time, substantially outperforming all competitors. This suggests the architectural choices (full attention), large scale (13B), and data curation were highly effective at capturing realistic dynamics.
*   **Text Alignment:** With a 61.8% win rate, it is highly competitive with the best model in this category (`CNTopA` at 62.6%), demonstrating its strong ability to follow complex prompts.
*   **Visual Quality:** The model scores a very high 95.7%, nearly on par with the top performers (`CNTopB` at 97.7% and `GEN-3` at 97.5%), indicating it generates aesthetically pleasing and high-fidelity videos.

## 6.2. Qualitative and Component Analysis

### 6.2.1. Qualitative Examples
The paper provides numerous visual examples to demonstrate the model's capabilities:
*   **Complex Scene Generation (Figure 12):** The model accurately generates a complex scene with multiple subjects (cat, man, child) and their interactions, showcasing strong text alignment.

    ![Figure 12: Prompt: A white cat sits on a white soft sofa like a person, while its long-haired male owner, with his hair tied up in a topknot, sits on the floor, gazing into the cat's eyes. His child stands nearby, observing the interaction between the cat and the man.](images/11.jpg)
    *该图像是四幅插图，展示了一只白色的猫坐在白色沙发上，旁边坐着它的主人，主人正在与猫进行互动，旁边还有一个小孩观察。这些图像呈现了人与宠物之间的亲密关系。*

*   **High-Motion Dynamics (Figure 14):** Examples like a roaring truck splashing mud demonstrate the model's excellence in capturing fast and dynamic motion.

    ![该图像是系列动态画面，展示了一辆越野车在泥地高速行驶时溅起的泥水。画面捕捉了车辆在不同角度和瞬间的运动状态，突显了其强劲的动力和灵活的操控能力。](images/14.jpg)
    *该图像是系列动态画面，展示了一辆越野车在泥地高速行驶时溅起的泥水。画面捕捉了车辆在不同角度和瞬间的运动状态，突显了其强劲的动力和灵活的操控能力。*

*   **Concept Generalization (Figure 15):** The model can generate fantastical scenes not seen in training, like an astronaut on a pink gemstone lake, proving its ability to compose novel concepts.

    ![Figure 15: HunyuanVideo's performance on concept generalization. The results of the three rows correspond to the text prompts (1) 'In a distant galaxy, an astronaut foats on a shimmering, pik, gemstone-like lak that re n color the undiy, eat sseThet nyri lak' rae, teo a hiper e plan et He ee uthisrti o hecol watr. Amae apture t hestrist playistrets (The night-blooming cactus fowers in the evening, with a bri, rapid closure.Time-lapse shot, extreme close-up. Realistic, Night lighting, Mysterious.' respectively.](images/19.jpg)
    *该图像是图表，展示了HunyuanVideo在概念泛化方面的表现。图中包含了三组生成结果，分别对应于不同的文本提示，通过色彩和场景设计展示了视频生成模型的创意表现。*

*   **Text Generation in Video (Figure 17):** `HunyuanVideo` can generate text that forms naturally within a scene, such as "WAKE UP" being spelled by sea foam, a challenging task requiring tight integration of semantics and visuals.

    ![Figure 17: High text-video alignment videos generated by HunyuanVideo. Top row: Prompt: A close-up of a wave crashing against the beach, the sea foam spells out "WAKE UP" on the sand. Bottom row: Prompt: In a garden filled with blooming flowers, "GROW LOVE" has been spelled out with colorful petals.](images/21.jpg)
    *该图像是展示HunyuanVideo生成的高文本视频对齐效果的插图。上排展示了海浪冲击沙滩，泡沫拼出"WAKE UP"字样的场景；下排为花园中色彩斑斓的花瓣拼出"GROW LOVE"字样。*

### 6.2.2. VAE Performance Analysis
The decision to train a 3D VAE from scratch is validated by its superior performance compared to other open-source VAEs.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th>Downsample Factor</th>
<th>|z|</th>
<th>ImageNet (256×256) PSNR↑</th>
<th>MCL-JCV (33×360×640) PSNR↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>FLUX-VAE [47]</td>
<td>1×8×8</td>
<td>16</td>
<td>32.70</td>
<td>-</td>
</tr>
<tr>
<td>OpenSora-1.2 [102]</td>
<td>4×8×8</td>
<td>4</td>
<td>28.11</td>
<td>30.15</td>
</tr>
<tr>
<td>CogvideoX-1.5 [93]</td>
<td>4× 8×8</td>
<td>16</td>
<td>31.73</td>
<td>33.22</td>
</tr>
<tr>
<td>Cosmos-VAE [64]</td>
<td>4×8×8</td>
<td>16</td>
<td>30.07</td>
<td>32.76</td>
</tr>
<tr>
<td>Ours</td>
<td>4×8×8</td>
<td>16</td>
<td><strong>33.14</strong></td>
<td><strong>35.39</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
The `HunyuanVideo` VAE achieves the highest PSNR on both image (`ImageNet`) and video (`MCL-JCV`) datasets. Its score of **35.39** on video reconstruction is significantly higher than the next best (`CogvideoX-1.5` at 33.22), highlighting the benefit of training a specialized video VAE from scratch with a well-designed loss function. Better reconstruction quality in the VAE leads to higher-fidelity final videos.

### 6.2.3. Downstream Application Demonstrations
The paper showcases the versatility of the pre-trained foundation model through several advanced applications:
*   **Video-to-Audio (V2A):** A module that generates synchronized sound effects and background music for a given video, using a similar DiT architecture trained on video-audio pairs.
*   **Image-to-Video (I2V):** The model is adapted to take a reference image as the first frame and generate a video that follows a text prompt, enabling animation from a static image.
*   **Avatar Animation (Figure 22):** A sophisticated application where the model can generate fully controllable talking avatars. It supports:
    *   **Audio-Driven Animation:** Generating upper-body animations with synchronized lip movements and expressions from an audio signal.
    *   **Pose-Driven Animation:** Driving a character's body motion using an input pose sequence.
    *   **Expression-Driven Animation:** Controlling facial expressions with an implicit expression representation.
    *   **Hybrid Control:** Combining pose and expression signals for fully controllable avatar generation.
        These applications demonstrate that `HunyuanVideo` is not just a text-to-video generator but a powerful visual foundation model.

        ![Figure 22: Overview of Avatar Animation built on top of HunyuanVideo. We adopt 3D VAE to encode and inject reference and pose condition, and use additional cross-attention layers to inject audio and expression signals. Masks are employed to explicitly guide where they are affecting.](images/26.jpg)
        *该图像是示意图，展示了基于 HunyuanVideo 的三种模型架构：T2V 模型、音频驱动模型和姿势与表情驱动模型。图中运用了多流 DIT 模块，结合适配器层，并通过不同输入（如噪声、音频、姿态）生成视频。相关模块及处理流程通过框架结构化呈现。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully introduces `HunyuanVideo`, a landmark achievement in the open-source community for video generation. The authors present a complete, systematic framework that demystifies the process of building a large-scale generative model. By integrating meticulous data curation, a highly scalable Transformer architecture, principled scaling laws, and an efficient training infrastructure, they developed a 13 billion parameter model that demonstrably rivals or exceeds the performance of leading closed-source systems. The key finding is that through a systematic and well-resourced effort, the performance gap between open and closed-source models can be bridged. The release of the model and its codebase represents a significant contribution, poised to accelerate research and innovation in the wider community.

## 7.2. Limitations & Future Work
The authors explicitly mention one area for future work, and several limitations can be inferred from the paper.

*   **Explicit Future Work:** The paper states that "the scaling property of progressive training from low-resolution to high-resolution will be left explored in future work." This suggests that while they used a progressive strategy, the optimal way to schedule data and resolutions for maximum efficiency is still an open question.
*   **Implicit Limitations:**
    *   **Video Duration:** The generated videos are still relatively short (5-6 seconds). Achieving long-term temporal coherence and narrative consistency over minutes remains a major challenge for all video generation models.
    *   **Physical Realism:** While motion quality is high, the model may still struggle with complex physical interactions, cause-and-effect reasoning, and object permanence over longer durations.
    *   **Replicability:** Although the core model is open-sourced, the full framework relies on proprietary assets, including the massive raw dataset, the in-house VLM for captioning, and the `Hunyuan-Large` model for prompt rewriting. This makes a full, 1-to-1 replication of their results by external parties challenging.
    *   **Evaluation Subjectivity:** The main results are based on human preference scores, which can be subjective and vary between evaluator groups. Broader, more objective benchmarks for video generation are still needed.

## 7.3. Personal Insights & Critique
`HunyuanVideo` is an impressive piece of engineering and a commendable contribution to the open-source AI community.

*   **Strengths and Inspirations:**
    *   **The Power of a Systematic Approach:** The paper's greatest strength is its emphasis on the entire ecosystem. It serves as a blueprint, showing that success in large-scale AI is not just about a clever architecture but about the synergy between data, model, and infrastructure.
    *   **Transparency in Scaling:** The inclusion of scaling law experiments is highly valuable. It provides a rare, public glimpse into the principled, data-driven decisions required to train billion-parameter models efficiently, a practice often hidden behind corporate walls.
    *   **Bridging the Gap:** This work is a powerful statement that the open-source community, with sufficient coordination and resources, can indeed keep pace with corporate giants. It will undoubtedly serve as a strong foundation for countless new projects and research directions.

*   **Critique and Potential Issues:**
    *   **Technical Report vs. Scientific Paper:** As a technical report, the paper focuses more on showcasing the system and its results rather than rigorous scientific investigation. For example, direct ablation studies comparing their MLLM text encoder against a standard T5, or their 3D RoPE against other positional encodings on the final model, would have further strengthened their architectural claims.
    *   **The "Secret Sauce" in Data:** The paper highlights the critical importance of data quality, yet the curation pipeline relies on internal tools and a dataset that cannot be shared. While the principles are explained, the community cannot directly build upon the exact data foundation that made `HunyuanVideo` successful. This is an inherent challenge in industry-led research but remains a barrier to full transparency and reproducibility.
    *   **Potential for Misuse:** Like all powerful generative models, `HunyuanVideo` has the potential to be used for creating realistic misinformation or deepfakes. While the authors don't delve into this, responsible deployment and watermarking strategies will be crucial as the technology proliferates.

        Overall, `HunyuanVideo` is a landmark paper that significantly pushes the open-source video generation frontier forward. Its value lies not only in the powerful model it delivers but also in the detailed, systematic framework it lays out for others to follow and build upon.