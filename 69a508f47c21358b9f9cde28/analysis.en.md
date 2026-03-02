# 1. Bibliographic Information

## 1.1. Title
SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model

## 1.2. Authors
The paper lists the "SkyReels Team" from "Skywork AI" as the author. A detailed contributor list is provided at the end of the paper, with Yahui Zhou as the Project Sponsor and Guibin Chen as the Project Leader. The contributors are affiliated with Skywork AI (Kunlun-inc), a technology company focused on large-scale AI models.

## 1.3. Journal/Conference
The paper is available on arXiv, which is a preprint server. This means it has not yet undergone formal peer review for publication in a specific academic journal or conference. The provided publication date suggests a future or hypothetical submission, as it is in the future.

## 1.4. Publication Year
The publication date listed on the paper is February 25, 2026. This is a future date, indicating that this is likely a placeholder or a target for a future release. The analysis will proceed based on the content as if it were a current publication.

## 1.5. Abstract
The abstract introduces SkyReels-V4 as a unified foundation model for jointly generating, inpainting, and editing video and audio. It highlights the model's key architectural features: a dual-stream Multimodal Diffusion Transformer (MMDiT) for separate video and audio synthesis, both guided by a shared Multimodal Large Language Model (MMLM) for understanding complex, multi-modal instructions (text, images, videos, masks, audio). For video tasks, it uses a channel-concatenation method to unify generation, inpainting, and editing. The model supports high-fidelity outputs up to 1080p resolution, 32 FPS, and 15-second durations. To achieve this efficiently, it employs a strategy of generating low-resolution full sequences alongside high-resolution keyframes, which are then upscaled using dedicated super-resolution and frame interpolation models. The abstract concludes by positioning SkyReels-V4 as the first model to combine multi-modal input, joint video-audio generation, and a unified framework for generation/inpainting/editing at a cinematic quality level.

## 1.6. Original Source Link
*   **Original Source Link:** `https://arxiv.org/abs/2602.21818`
*   **PDF Link:** `https://arxiv.org/pdf/2602.21818v2`
*   **Publication Status:** The paper is a preprint on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
*   **Core Problem:** The field of generative AI has seen rapid progress in video generation, but existing models are often fragmented. Models that excel at video synthesis may not handle audio, those that generate audio and video together may not accept complex multi-modal inputs (like reference images or videos), and those that do may not offer comprehensive editing and inpainting capabilities within the same framework. This fragmentation leads to complex, multi-step workflows, potential audio-visual asynchrony, and a lack of unified control.
*   **Importance and Gaps:** Creating compelling, immersive media requires the seamless synergy of visuals and sound. Previous models often treated video and audio generation as separate tasks (`T2V` then `V2A`), leading to mismatches. While recent models have started to tackle joint audio-visual generation, they typically lack one or more of the following:
    1.  **Unified Multi-modal Conditioning:** The ability to simultaneously understand and act on instructions combining text, reference images, video clips, masks, and audio samples.
    2.  **Integrated Generation and Editing:** A single model that can perform text-to-video generation, image-to-video animation, video extension, and fine-grained editing (inpainting) through a consistent interface.
    3.  **High-Fidelity and Efficiency:** Generating long-duration (15s+), high-resolution (1080p) videos with synchronized audio is computationally prohibitive for most architectures.
*   **Innovative Idea:** The paper's central idea is to create a **single, unified foundation model** that addresses all these gaps. The innovation lies in its architecture and training strategy:
    1.  **Dual-Stream MMDiT:** Separating video and audio synthesis into two specialized but interconnected transformer streams allows for high-quality generation in both modalities while enabling synchronization through cross-attention.
    2.  **Shared MLLM Encoder:** Using a powerful MLLM as a shared "brain" to interpret diverse multi-modal prompts provides a unified, semantically rich conditioning signal for both streams.
    3.  **Unified Inpainting Framework:** A clever channel-concatenation technique treats generation, editing, and inpainting as variations of the same core task, simplifying the model's design and usage.
    4.  **Efficiency Strategy:** A practical approach to high-resolution generation by producing a low-resolution draft and high-resolution keyframes, then upscaling, which balances quality and computational cost.

## 2.2. Main Contributions / Findings
*   **Primary Contributions:**
    1.  **SkyReels-V4 Model:** The introduction of a dual-stream MMDiT-based foundation model that performs **joint video and audio generation** under complex **multi-modal conditioning**.
    2.  **Unified Video Inpainting Framework:** A novel channel-concatenation approach that unifies diverse tasks like image-to-video, video extension, and region-based editing into a single inpainting formulation, which also naturally supports vision-referenced editing.
    3.  **Efficient High-Resolution Generation Strategy:** A practical method for achieving 1080p, 32 FPS, 15-second video generation by jointly producing low-resolution full sequences and high-resolution keyframes, followed by dedicated super-resolution and frame interpolation models.
*   **Key Findings:**
    1.  The proposed architecture is effective at unifying a wide range of video creation tasks that were previously handled by separate, specialized models.
    2.  The model demonstrates state-of-the-art performance, ranking second in the Artificial Analysis Arena leaderboard and outperforming other leading commercial systems (like Veo 3.1, Kling 2.6) in comprehensive human evaluations (SkyReels-VABench) across dimensions like instruction following, motion quality, and overall quality.
    3.  The paper claims SkyReels-V4 is the first model to successfully integrate multi-modal input, joint video-audio generation, and a full suite of generation/inpainting/editing capabilities at cinematic quality, setting a new benchmark for multi-modal video foundation models.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
*   **Diffusion Models:** A class of generative models that learn to create data by reversing a gradual noising process. The process starts with a real data sample (e.g., an image) and progressively adds Gaussian noise over many steps until it becomes pure noise. The model is then trained to predict and remove the noise at each step. To generate a new sample, the model starts with random noise and iteratively "denoises" it, guided by a condition (like a text prompt), until a clean data sample is formed.
*   **Transformers:** A neural network architecture originally designed for natural language processing tasks, which relies heavily on the `self-attention` mechanism. Unlike recurrent neural networks (RNNs) that process data sequentially, transformers can process all input tokens (e.g., words in a sentence or patches of an image) simultaneously. This parallel processing and the ability to weigh the importance of different parts of the input make them extremely powerful and scalable.
*   **Diffusion Transformer (DiT):** An architecture that replaces the commonly used U-Net backbone in diffusion models with a Transformer. In a `DiT`, the noisy input (e.g., a noisy image latent) is broken down into a sequence of patches or tokens. The Transformer then operates on this sequence to predict the noise, treating the generation process as a sequence-to-sequence problem. This architecture has been shown to be highly scalable and effective for high-resolution image and video generation.
*   **Multimodal Large Language Models (MMLMs):** These are large language models that have been extended to understand and process information from multiple modalities, not just text. They can take inputs that combine text, images, and sometimes audio or video, and generate text-based responses or embeddings that reflect a deep, cross-modal understanding. In SkyReels-V4, an MLLM is used as a powerful text encoder that can interpret prompts containing both text and references to images or videos.
*   **Variational Autoencoder (VAE):** A type of neural network used for unsupervised learning and dimensionality reduction. It consists of two parts: an **encoder** that compresses the input data (e.g., a high-resolution image) into a smaller, low-dimensional latent representation, and a **decoder** that reconstructs the original data from this latent representation. VAEs are commonly used in diffusion models to work in a compressed latent space, which is much more computationally efficient than working with raw pixel data.

## 3.2. Previous Works
*   **Early Video Diffusion Models:** Initial approaches like `Video Diffusion Models` [21] and `AnimateDiff` [22] often used $2D + 1D$ architectures. This means they adapted pre-trained 2D image diffusion models (like Stable Diffusion) for video by adding temporal modules (the `1D` part) to learn motion. While effective, they could struggle with long-term temporal consistency.
*   **DiT-based Video Models:** The success of `DiT` [23] in image generation led to its adoption in video. `Sora` [24] by OpenAI was a landmark model that demonstrated the power of large-scale training on a `DiT`-based architecture with spatiotemporal attention, achieving unprecedented video quality and consistency. Many subsequent models, including SkyReels-V4, build on this paradigm.
*   **Joint Audio-Video Generation:** Early work like `MM-Diffusion` [38] used coupled U-Nets for audio and video. More recent models have adopted `DiT`-based architectures. `AV-DiT` [39] uses an adapter-based approach, while others use dual-stream architectures like SkyReels-V4. For instance, `LTX-2` [36] proposes asymmetric streams for efficiency, and `Apollo` [45] uses a unified single-tower model. These works aim to solve the critical problem of audio-visual synchronization.
*   **Multimodal-Referenced Generation:** Models have evolved beyond simple text-to-video. `Vidu` [7] introduced reference-to-video generation from multiple images. `RunwayAleph` [8] showcased advanced in-context video editing. `Kling-Omni` [16] was among the first to support both image and video references for video generation, but it lacked native audio synthesis. SkyReels-V4 aims to unify these reference-based capabilities with joint audio-video generation.

## 3.3. Technological Evolution
The field of video generation has evolved rapidly through several key stages:
1.  **GANs to Diffusion Models:** Early video generation was dominated by Generative Adversarial Networks (GANs), which struggled with training stability and mode collapse. Diffusion models emerged as a more stable and powerful alternative, leading to higher-quality results.
2.  **2D to 3D/Spatiotemporal Architectures:** Initial video diffusion models were often extensions of 2D image models. The trend shifted towards architectures that treat video as a holistic spatiotemporal volume, using 3D convolutions or spatiotemporal attention mechanisms within Transformers (`DiT`), as seen in `Sora` and SkyReels-V4.
3.  **Unimodal to Multimodal:** The focus has shifted from text-only conditioning (`T2V`) to accepting a rich variety of inputs, including images (`I2V`), videos (`V2V`), and audio (`A2V`).
4.  **Separate to Joint Synthesis:** The paradigm has moved from generating video and audio in separate pipelines to **joint audio-visual generation** within a single model, ensuring better synchronization and coherence.
5.  **Generation to Unified Creation:** The most recent evolution, which SkyReels-V4 represents, is the move from models that only *generate* content to unified platforms that can **generate, edit, and inpaint** video and audio through a single, consistent interface.

    SkyReels-V4 fits into the latest stage of this evolution, aiming to be a comprehensive, all-in-one foundation model for multi-modal video and audio creation.

## 3.4. Differentiation Analysis
Compared to related works, SkyReels-V4's core innovations are:
*   **Unification:** Its primary differentiator is its **unification of capabilities**. While other models might excel in one or two areas (e.g., `Kling-Omni` in visual referencing, `LTX-2` in efficient AV generation), SkyReels-V4 is presented as the first to combine **all five** key features in one model:
    1.  Rich multi-modal inputs (text, image, video, mask, audio).
    2.  Joint video-audio generation.
    3.  A unified inpainting/editing framework.
    4.  High-resolution, long-duration output.
    5.  Computational efficiency via a specialized generation strategy.
*   **Architectural Synergy:** The combination of a **dual-stream MMDiT** with a **shared MLLM encoder** is a powerful design choice. The MLLM provides high-level semantic understanding across all input types, while the dual streams allow for specialized, high-quality synthesis in their respective domains (video and audio), with tight synchronization enforced by bidirectional cross-attention.
*   **Inpainting as a General Interface:** The use of **channel concatenation** to handle all video manipulation tasks is a simple yet elegant solution. It reframes complex operations like I2V, video extension, and editing as simple variations of inpainting, controlled by a mask. This simplifies both the model architecture and the user experience.

# 4. Methodology

## 4.1. Principles
The core principle of SkyReels-V4 is to create a unified and powerful generative model by combining specialized components within a synergistic architecture. The model is built on a **dual-stream Multimodal Diffusion Transformer (MMDiT)**. One stream is dedicated to synthesizing video, while the other generates temporally aligned audio. This separation allows each stream to be optimized for its specific modality. Synchronization is achieved through **bidirectional cross-attention** layers that allow the audio and video streams to exchange information throughout the generation process.

To handle diverse and complex user instructions, both streams are conditioned by a single, powerful **Multimodal Large Language Model (MLLM)**, which acts as a shared text encoder. This allows the model to interpret prompts that mix text with references to images, videos, and audio.

For video editing and manipulation, the model employs a unified **inpainting framework via channel concatenation**. This elegant approach treats tasks like image-to-video, video extension, and region-based editing as variations of a single inpainting problem, controlled by a spatiotemporal mask.

Finally, to make high-resolution, long-duration generation practical, the model uses an efficient two-stage process: first, it generates a low-resolution full video and high-resolution keyframes simultaneously, and then dedicated models perform super-resolution and frame interpolation.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Dual-Stream MMDiT Architecture for Joint Video-Audio Generation
The foundation of SkyReels-V4 is a dual-stream `MMDiT`. One branch is a pre-trained text-to-video model, and the other is an audio-branch transformer trained from scratch with a matching architecture.

*   **Hybrid Dual-Stream and Single-Stream Blocks:** The `MMDiT` architecture uses a hybrid design for its transformer blocks to balance performance and efficiency.
    *   **Dual-Stream Layers (Initial $M$ layers):** In the early layers, video/audio tokens and text tokens are processed by separate parameters (for layer normalization, QKV projections, MLPs). They only interact during the joint self-attention step. This allows each modality to learn specialized features initially.
    *   **Single-Stream Layers (Subsequent $N$ layers):** In the later layers, the video/audio and text tokens share parameters, which reduces the model size and improves computational efficiency. This hybrid approach is claimed to achieve faster convergence.

        The joint self-attention mechanism in the dual-stream layers is formulated as follows. First, the Query (Q), Key (K), and Value (V) projections are computed separately for the video/audio tokens ($ \mathbf{x}_v $) and text tokens ($ \mathbf{x}_t $):
    \$
    \begin{array} { r l } & { \mathbf { Q } _ { v } , \mathbf { K } _ { v } , \mathbf { V } _ { v } = \mathbf { Q } \mathbf { K } \mathbf { V } _ { v } ( \mathrm { LayerNorm } _ { v } ( \mathbf { x } _ { v } ) ) , } \\ & { \mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } = \mathbf { Q } \mathbf { K } \mathbf { V } _ { t } ( \mathrm { LayerNorm } _ { t } ( \mathbf { x } _ { t } ) ) , } \end{array}
    \$
    Here, $\mathbf{x}_v$ and $\mathbf{x}_t$ are the input token embeddings for video/audio and text respectively. $\mathrm{LayerNorm}$ is layer normalization, and $\mathbf{QKV}$ represents the linear projection layers. The projections are then concatenated and fed into a single attention function:
    \$
    \mathbf { x } _ { v } ^ { \prime } , \mathbf { x } _ { t } ^ { \prime } = \mathbf { A } \mathrm { t t e n t i o n } ( [ \mathbf { Q } _ { v } ; \mathbf { Q } _ { t } ] , [ \mathbf { K } _ { v } ; \mathbf { K } _ { t } ] , [ \mathbf { V } _ { v } ; \mathbf { V } _ { t } ] ) ,
    \$
    where $[ \cdot ; \cdot ]$ denotes concatenation, and $\mathbf{x}_v^\prime, \mathbf{x}_t^\prime$ are the updated tokens.

*   **Reinforced Text Conditioning via Cross-Attention:** To prevent the influence of the text prompt from fading in the later, single-stream layers, an additional cross-attention layer is added after the self-attention block. This layer re-injects the original text information.
    \$
    \mathbf { x } _ { v } ^ { \prime \prime } = \mathbf { x } _ { v } ^ { \prime } + \mathrm { A t t e n t i o n } ( \mathbf { Q } = \mathbf { x } _ { v } ^ { \prime } , \mathbf { K } = \mathbf { x } _ { t } , \mathbf { V } = \mathbf { x } _ { t } ) ,
    \$
    Here, the updated video/audio tokens $\mathbf{x}_v^\prime$ act as queries, while the original text tokens $\mathbf{x}_t$ serve as keys and values. This ensures strong semantic guidance throughout the network.

*   **Bidirectional Audio-Video Cross-Attention:** To ensure tight temporal synchronization, each transformer block contains a pair of cross-attention layers where the audio and video streams attend to each other.
    \$
    \begin{array} { r } { { \bf a } _ { i } ^ { \prime } = { \bf a } _ { i } + { \bf C } \mathrm { r o s s A t t n } ( { \bf Q } = { \bf a } _ { i } , { \bf K } = { \bf v } _ { i } , { \bf V } = { \bf v } _ { i } ) , } \\ { { \bf v } _ { i } ^ { \prime \prime } = { \bf v } _ { i } ^ { \prime } + { \bf C } \mathrm { r o s s A t t n } ( { \bf Q } = { \bf v } _ { i } ^ { \prime } , { \bf K } = { \bf a } _ { i } ^ { \prime } , { \bf V } = { \bf a } _ { i } ^ { \prime } ) , } \end{array}
    \$
    where $\mathbf{a}_i$ and $\mathbf{v}_i$ are the audio and video features at layer $i$. First, the audio features attend to the video features. Then, the updated video features attend to the newly updated audio features. This bidirectional flow of information allows the model to learn and enforce audio-visual correspondences.

*   **Temporal Alignment via RoPE Scaling:** The model faces a challenge where the temporal resolutions of video and audio latents are different (e.g., 21 video frames vs. 218 audio tokens for a 5s clip). To align them, the model uses Rotary Positional Embeddings (`RoPE`) and scales the frequencies of the audio `RoPE` by a factor of $21 / 218 \approx 0.09633$. This aligns the temporal scales, helping the model learn consistent correspondences.

*   **Training Objective (Flow Matching):** The model is trained using a flow matching objective. Given clean video latents $\mathbf{z}_v^0$ and audio latents $\mathbf{z}_a^0$, a timestep $t$ is sampled, and noisy latents are created: $\mathbf { z } _ { v } ^ { t } = t \mathbf { z } _ { v } ^ { 0 } + ( 1 - t ) \mathbf { \epsilon } _ { v }$ and $\mathbf { z } _ { a } ^ { t } = t \mathbf { z } _ { a } ^ { \tilde { 0 } } + ( 1 - t ) \epsilon _ { a }$, where $\epsilon$ is random noise. The model is trained to predict the "velocity" field $\mathbf{v}_\theta$ that points from the noisy sample towards the clean sample. The loss function is:
    \$
    \mathcal { L } _ { \mathrm { f l o w } } = \mathbb { E } _ { t , z _ { v } ^ { 0 } , z _ { a } ^ { 0 } , \epsilon _ { v } , \epsilon _ { a } } \left[ \left\| \mathbf { v } _ { \theta } ^ { v } ( t , \mathbf { z } _ { v } ^ { t } , \mathbf { z } _ { a } ^ { t } , \mathbf { c } ) - ( \mathbf { z } _ { v } ^ { 0 } - \epsilon _ { v } ) \right\| ^ { 2 } + \left\| \mathbf { v } _ { \theta } ^ { a } ( t , \mathbf { z } _ { a } ^ { t } , \mathbf { z } _ { v } ^ { t } , \mathbf { c } ) - ( \mathbf { z } _ { a } ^ { 0 } - \epsilon _ { a } ) \right\| ^ { 2 } \right] ,
    \$
    where $\mathbf{c}$ is the conditioning information (from the MLLM). This joint loss encourages both streams to learn synchronized features.

### 4.2.2. Unified Video Inpainting via Channel Concatenation
To handle a wide array of video manipulation tasks in a unified manner, the model concatenates the noisy video latent, the conditional frames, and a spatiotemporal mask along the channel dimension.
\$
{ \bf Z } _ { \mathrm { i n p u t } } = \mathrm { C o n c a t } ( { \bf V } , { \bf I } , { \bf M } ) ,
\$
*   $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times C}$: The noisy video latent being denoised.
*   $\mathbf{I} \in \mathbb{R}^{T \times H \times W \times C}$: The VAE-encoded known/conditional frames.
*   $\mathbf{M} \in \mathbb{R}^{T \times H \times W \times 1}$: A binary mask where `1` indicates known regions (to be preserved) and `0` indicates unknown regions (to be generated).

    By changing the mask $\mathbf{M}$ and conditional frames $\mathbf{I}$, this single formulation can handle various tasks:
*   **Text-to-Video (T2V):** $\mathbf{M}$ is all zeros. No conditional frames are provided.
*   **Image-to-Video (I2V):** The first frame of $\mathbf{I}$ is the input image, and the mask $\mathbf{M}$ is `1` for the first timestep ($t=0$) and `0` for all subsequent timesteps.
*   **Video Extension:** The mask $\mathbf{M}$ is `1` for the initial known frames ($t < k$) and `0` for the frames to be generated ($t \geq k$).
*   **Video Editing:** The mask $\mathbf{M}$ is `1` for all pixels that should be preserved and `0` for the spatiotemporal regions that need to be edited or inpainted.

### 4.2.3. Multi-Modal In-Context Learning for Vision-Referenced Generation
The model supports advanced conditioning using reference images and videos.

*   **Multi-Modal Instruction Following with MLLM:** Reference inputs (images, videos) are processed jointly with the text prompt by the MLLM encoder. This produces semantically rich embeddings that capture the combined meaning of the instructions (e.g., "make a video of person in $@image_1$ doing the dance from $@video_1$").
*   **In-Context Visual Conditioning via Self-Attention:** To provide direct visual guidance, the reference images/videos are also encoded by the VAE into latent tokens $\mathbf{Z}_{\mathrm{cond}}$. These tokens are then prepended to the noisy video latents $\mathbf{Z}_{\mathrm{video}}$ before being fed into the self-attention layers of the `MMDiT`.
    \$
    \mathbf { Z } _ { \mathrm { a t t n } } = [ \mathbf { Z } _ { \mathrm { c o n d } } ; \mathbf { Z } _ { \mathrm { v i d e o } } ] ,
    \$
    This allows the model to directly "see" the visual details of the reference material during the generation process, enabling it to copy styles, appearances, or motions.

*   **Temporal Positional Disambiguation:** To help the model distinguish between conditioning tokens and the video tokens being generated, they are assigned different temporal positions using an offset 3D `RoPE`. The conditioning latents are given negative temporal indices, while the video latents are given standard positive indices.
    \$
    \mathrm { R o P E } _ { \mathrm { t e m p o r a l } } ( \mathbf { Z } _ { \mathrm { c o n d } , i } ) = \mathrm { R o P E } ( t = - N _ { \mathrm { c o n d } } + i ) , \quad \mathrm { R o P E } _ { \mathrm { t e m p o r a l } } ( \mathbf { Z } _ { \mathrm { v i d e o } , j } ) = \mathrm { R o P E } ( t = j ) ,
    \$
    where $N_{\mathrm{cond}}$ is the number of condition tokens. This offset provides a clear signal to the model about which tokens are fixed references and which are part of the sequence to be generated.

### 4.2.4. Video Super-Resolution and Frame Interpolation (Refiner)
To achieve high-resolution (1080p) and high-framerate (32 FPS) output efficiently, the system uses a two-stage approach. The base `MMDiT` model generates a low-resolution full video sequence and high-resolution keyframes. A dedicated **Refiner** model then takes these outputs to produce the final high-quality video.
*   **Architecture:** The Refiner is initialized with the weights of the pre-trained video generation model. It takes the low-resolution video frames (upsampled), the high-resolution keyframes, and the text prompt as input. These are concatenated along the channel dimension and fed into the Refiner's `DiT` architecture.
*   **Computational Efficiency:** The Refiner uses **Video Sparse Attention (VSA)** [59], a trainable sparse attention mechanism that reduces the computational cost of processing long, high-resolution sequences by approximately 3x without sacrificing quality.

# 5. Experimental Setup

## 5.1. Datasets
The paper describes an extensive data pipeline for training SkyReels-V4, using both real-world and synthetic data across image, video, and audio modalities.

*   **Data Sources:**
    *   **Real-World Data:**
        *   **Images:** Public datasets like LAION [46] and Flickr [47].
        *   **Videos:** Public datasets such as WebVid-10M [48], Koala-36M [49], and OpenHumanVid [50]. The paper also mentions using licensed data from movies, TV series, and short videos.
        *   **Audio:** Public datasets including Emilia [51], AudioSet [52], VGGSound [53], and SoundNet [54].
    *   **Synthetic Data:** The authors generated synthetic data to cover scenarios underrepresented in real-world datasets. This includes:
        *   **In-video text:** For learning to render text within videos.
        *   **Multilingual speech:** Using Text-to-Speech (TTS) models to ensure broad language coverage.
        *   **Inpainting/Editing data:** Since high-quality paired data for inpainting is rare, they constructed a synthetic dataset using segmentation models and other generative techniques.

*   **Data Processing and Captioning:**
    *   A rigorous filtering pipeline was used to ensure data quality, checking for aesthetics, technical quality (e.g., Signal-to-Noise Ratio for audio), and content relevance.
    *   For videos with audio, SyncNet [58] was used to ensure audio-visual synchronization, retaining only clips with high confidence and low offset.
    *   A structured captioning format was developed. Captions are not just simple descriptions but follow a standardized order and use special tokens to denote different audio events: $<text>$, $<sfx>$, $<dialogue>$, $<singing>$, and $<bgm>$.

*   **Reason for Dataset Choices:** The massive scale and diversity of the chosen datasets are crucial for training a powerful foundation model. The combination of public, licensed, and synthetic data ensures broad coverage of concepts, styles, languages, and tasks, which is essential for the model's generalization capabilities.

## 5.2. Evaluation Metrics
The paper primarily relies on human evaluation for performance assessment, which is common for generative models where objective metrics often fail to capture perceptual quality.

### 5.2.1. Elo Rating
*   **Conceptual Definition:** The Elo rating system is a method for calculating the relative skill levels of players in competitor-vs-competitor games. In the context of generative models, it is used to rank models based on pairwise human comparisons. Users are shown outputs from two different models for the same prompt and are asked to choose which one is better. Each "win" for a model increases its Elo score, while a "loss" decreases it. A higher Elo score indicates a higher user preference.
*   **Mathematical Formula:** The expected score of player A against player B is given by:
    \$
    E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
    \$
    The new rating for player A is then updated based on the actual score $S_A$ (1 for a win, 0.5 for a draw, 0 for a loss):
    \$
    R'_A = R_A + K(S_A - E_A)
    \$
*   **Symbol Explanation:**
    *   $R_A, R_B$: Current Elo ratings of players A and B.
    *   $E_A$: Expected score for player A.
    *   `R'_A`: New rating for player A after the match.
    *   $K$: A constant that determines the maximum possible adjustment per game (typically between 10 and 40).

### 5.2.2. Human Evaluation on SkyReels-VABench
The authors created a custom benchmark, **SkyReels-VABench**, with over 2000 prompts. Professional evaluators assessed the generated videos along five key dimensions using two methods:

*   **Absolute Scoring (Likert Scale):**
    *   **Conceptual Definition:** Evaluators rate the quality of a single video on a scale from 1 to 5 for various criteria (e.g., Visual Quality, Motion Quality). This provides a direct, absolute measure of performance.
    *   **Scale:** 1 = Extremely Dissatisfied, 2 = Dissatisfied, 3 = Neutral, 4 = Satisfied, 5 = Extremely Satisfied.

*   **Good-Same-Bad (GSB) Comparison:**
    *   **Conceptual Definition:** A pairwise comparison where evaluators are shown the outputs from SkyReels-V4 and a baseline model for the same prompt. They must decide if the SkyReels-V4 output is "Good" (better), "Same" (comparable), or "Bad" (worse) than the baseline. This provides a more direct and granular measure of relative performance.

## 5.3. Baselines
The paper compares SkyReels-V4 against several state-of-the-art proprietary commercial video-audio generation systems:
*   **Veo 3.1 (Google)**
*   **Kling 2.6 (Kuaishou)**
*   **Seedance 1.5 Pro (ByteDance)**
*   **Wan 2.6 (Alibaba)**

    These baselines were chosen because they represent the leading models in the field of joint video-audio generation at the time, making them strong competitors for assessing state-of-the-art performance.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Artificial Analysis Arena
SkyReels-V4 was evaluated on the Artificial Analysis leaderboard for "text-to-video with audio" generation. As shown in Figure 3 from the paper, the model achieved the **second rank** among all participating systems, with an Elo score of 1090. This is a very strong result, demonstrating that the model's output quality is highly competitive and preferred by the general public when compared against other leading models like Veo 3.1, Sora-2, and Kling 3.0.

The following figure (Figure 3 from the original paper) shows the leaderboard ranking.

![Figure 3:Artificial Analysis Text-to-Video with Audio Arena Leaderboard. Our model ranks second among all competing baselines including Veo 3.1, grok-imagine-vide, Sora-2, Vidu-Q3, Wan 2.6 and etc.](images/3.jpg)
*该图像是一个图表，展示了SkyReels V4在文本到视频生成的音频领域中的排名。该模型在众多竞争基线中排名第二，包括Veo 3.1等，ELO得分为1,090。*

### 6.1.2. Human Assessments on SkyReels-VABench
The paper provides a more detailed breakdown of performance through its custom benchmark, SkyReels-VABench.

*   **Absolute Scoring Results:** Figure 4 presents the average scores on the 5-point Likert scale across five dimensions. SkyReels-V4 achieves the **highest overall average score** compared to all baselines. The breakdown reveals specific strengths:
    *   **Instruction Following & Motion Quality:** SkyReels-V4 shows a clear lead in these areas, suggesting its MLLM encoder and temporal modeling are highly effective.
    *   **Visual Quality:** It performs on par with the strongest competitors.
    *   **Audio-Visual Synchronization & Audio Quality:** While the advantage is more modest, it still maintains state-of-the-art performance.

        The following figure (Figure 4 from the original paper) shows the absolute scoring results.

        ![Figure 4:Absolute scoring results (5-point Likert scale comparing SkyReels V4 against baselines. Higher score indicate better performance.](images/4.jpg)
        *该图像是一个图表，展示了SkyReels V4与多个基线模型的评分结果（5点Likert量表），涵盖整体质量、指令遵循、视听同步、视觉质量、动作质量和音频质量等指标。图中较高的评分表明了SkyReels V4在这些方面的优越性能。*

*   **Good-Same-Bad (GSB) Comparison:** The pairwise GSB comparisons provide further evidence of SkyReels-V4's superiority.
    *   **Overall Quality:** Figure 5 shows that when compared against all baselines combined, SkyReels-V4 receives a significantly higher proportion of "Good" ratings than "Bad" ratings, indicating a strong overall preference.
    *   **Per-Dimension, Per-Baseline Comparison:** Figures 6, 7, 8, and 9 (in the original paper's appendix, which seems to have been aggregated into a single figure in the main text) show detailed pairwise comparisons against each baseline. SkyReels-V4 consistently outperforms Kling 2.6, Seedance 1.5 Pro, Veo 3.1, and Wan 2.6 across most, if not all, of the five evaluation dimensions. The green bars (preference for SkyReels-V4) are consistently larger than the orange bars (preference for the competitor).

        The following figure (Figure 5 from the original paper) shows the overall GSB quality comparison.

        ![该图像是一个条形图，展示了SkyReels V4与其他模型（Kling 2.6、Veo 3.1、Seedance 1.5 Pro、Wan2.6）在总体质量上的偏好比较。数据表明，SkyReels V4在所有比较中均表现较好。](images/5.jpg)
        *该图像是一个条形图，展示了SkyReels V4与其他模型（Kling 2.6、Veo 3.1、Seedance 1.5 Pro、Wan2.6）在总体质量上的偏好比较。数据表明，SkyReels V4在所有比较中均表现较好。*

The following figures (Figures 6, 7, 8, 9, combined into one composite image in the PDF) show the per-dimension GSB comparisons against each baseline.

![该图像是条形图，展示了 SkyReels V4 与竞争对手 Kling 2.6 在多个质量维度上的偏好比较，包括指令遵循、音视同步、视觉质量、动作质量和音频质量。各项指标上，SkyReels V4 的偏好较高，显示了其在多模态生成中的优势。](images/6.jpg)![](images/7.jpg)![](images/8.jpg)![](images/9.jpg)
*该图像是条形图，展示了 SkyReels V4 与竞争对手 Kling 2.6 在多个质量维度上的偏好比较，包括指令遵循、音视同步、视觉质量、动作质量和音频质量。各项指标上，SkyReels V4 的偏好较高，显示了其在多模态生成中的优势。*

The collective results from both the public arena and the controlled human evaluation strongly support the paper's claim that SkyReels-V4 achieves state-of-the-art performance, demonstrating a robust and high-quality solution for joint video-audio generation.

## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper, summarizing the multi-stage training strategy:

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Resolution</td><td rowspan=1 colspan=1>Data Volume</td><td rowspan=1 colspan=1>Epochs</td></tr><tr><td rowspan=1 colspan=1></td><td colspan=4>Video Pretrain</td></tr><tr><td rowspan=1 colspan=1>T2I</td><td rowspan=1 colspan=1>Stage 1</td><td rowspan=1 colspan=1>256px</td><td rowspan=1 colspan=1>3B images</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2I + T2V</td><td rowspan=1 colspan=1>Stage 2</td><td rowspan=1 colspan=1>256px, 16fps, 2-10s</td><td rowspan=1 colspan=1>1B images / 400M videos</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2I + T2V + Inpaint(Image Inpaint, I2V, V2V, Edit)</td><td rowspan=1 colspan=1>Stage 3</td><td rowspan=1 colspan=1>256px, 16fps, 2-15s(Inpaint: 5% each)</td><td rowspan=1>1B images / 400M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Mixed Tasks(T2I, T2V, Inpaint)</td><td rowspan=1 colspan=1>Stage 4</td><td rowspan=1 colspan=1>256/480px, 16fps, 2-15s(Inpaint ratio unchanged)</td><td rowspan=1 colspan=1>100M images / 100M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Mixed Tasks(T2I, T2V, Inpaint)</td><td rowspan=1 colspan=1>Stage 5</td><td rowspan=1 colspan=1>480/720/1080px,16fps, 3-15s</td><td rowspan=1 colspan=1>50M images / 50M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Multi-modal Condition(Image/Video Ref: 20% each)(T2V: 60%)</td><td rowspan=1 colspan=1>Stage 6</td><td rowspan=1>480/720/1080px,16fps, 3-15s</td><td rowspan=1 colspan=1>20M images / 50M videos</td><td rowspan=1 colspan=1>2</td></tr><tr><td colspan=4>Audio Pretrain</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Audio Backbone</td><td colspan=2>Pretrain        Variable length, up to 15s</td><td rowspan=1 colspan=1>Hundreds of thousands of hours</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1></td><td colspan=2>Video-Audio Joint Training</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T2V + T2AV + T2A</td><td rowspan=1 colspan=1>Joint Pretrain</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>50% video data + T2A data</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td colspan=2>Video-Audio Supervised Fine-tuning</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T2AV + Multi-modal</td><td rowspan=1 colspan=1>SFT Stage 1</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>5M videos (Multi-modal: 20%)</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>T2AV + Multi-modal</td><td rowspan=1 colspan=1>SFT Stage 2</td><td rowspan=1 colspan=1>720/1080px, 16fps, 5-15s</td><td rowspan=1 colspan=1>1M curated videos</td><td rowspan=1 colspan=1>3</td></tr></table>

The following are the results from Table 2 of the original paper, detailing the dimensions for human evaluation:

<table><tr><td>Dimension</td><td>Sub-dimension</td><td>Evaluation Criteria</td></tr><tr><td rowspan="2">Instruction Follow- ing</td><td>Video Instruction Following Subject description Subject interaction Camera movement</td><td>Accurate representation of subjects, attributes, and appearances Correct execution of actions, interactions, and motion dynamics Proper execution of camera operations (pan, tilt, zoom, dolly)</td></tr><tr><td>Style and aesthetics Multi-shot consistency Audio Instruction Following Semantic adherence</td><td>Adherence to visual styles, color palettes, and artistic directions Correct shot transitions, cross-shot coherence, and reference accuracy Fidelity to audio content and characteristics</td></tr><tr><td>Audio-Visual Syn-</td><td>Lip-sync accuracy Sound effect alignment Atmospheric matching</td><td>accuracy Precise speech-mouth synchronization and correct speaker identification Temporal correspondence between visual events and sound effects Coherence between BGM, scene atmosphere, and emotional tone</td></tr><tr><td>Visual Quality</td><td>Visual clarity Color accuracy Compositional quality Structural integrity</td><td>Sharpness, definition, and resolution Natural color balance and saturation without distortion Aesthetic composition, framing, and visual balance Absence of visual artifacts and corruptions</td></tr><tr><td>Motion Quality</td><td>Physical plausibility Motion fluidity Motion stability Temporal consistency Motion vividness</td><td>Adherence to physical laws (gravity, inertia, momentum) Smooth transitions without abrupt discontinuities Absence of jittering, deformation, and flickering Consistency of dynamic elements across frames Action, camera, atmospheric, and emotional expressiveness</td></tr><tr><td>Audio Quality</td><td>Absence of artifacts Spatial soundstage Timbre realism Signal clarity Dynamic range</td><td>No clipping, truncation, distortion, or glitches Appropriate stereo imaging and spatial rendering Natural and realistic tonal qualities Clean audio with appropriate signal-to-noise ratio Appropriate audio level variation without compression artifacts</td></tr></table>

## 6.3. Ablation Studies / Parameter Analysis
The paper does not include a dedicated section for ablation studies. While the multi-stage training strategy implicitly suggests the importance of each stage (e.g., starting with image pre-training, then adding video, then inpainting), there are no controlled experiments that isolate and remove specific components of the final model (e.g., removing bidirectional cross-attention, using a single-stream architecture, or removing the MLLM encoder) to quantify their individual contributions to the overall performance. This is a common omission in technical reports for large-scale models but would be a valuable addition for a formal academic publication.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper introduces SkyReels-V4, a comprehensive, unified foundation model for multi-modal video and audio creation. Its core contribution is the successful integration of several critical capabilities into a single framework: joint video-audio generation, rich multi-modal conditioning, and a full suite of generation, inpainting, and editing functions. The model's dual-stream `MMDiT` architecture, guided by a shared MLLM, and its elegant channel-concatenation method for video manipulation enable it to produce high-fidelity, synchronized audio-visual content at up to 1080p resolution. Extensive evaluations on a public arena and a custom benchmark demonstrate that SkyReels-V4 achieves state-of-the-art performance, outperforming leading commercial systems. The authors position the work as a new baseline for future research in multi-modal generative AI.

## 7.2. Limitations & Future Work
The paper itself does not explicitly state its limitations or outline specific future work. However, based on the content and the state of the field, several potential areas can be inferred:

*   **Potential Limitations:**
    *   **Physical Realism:** While motion quality is a strong point, complex physical interactions (e.g., fluid dynamics, soft-body collisions) likely remain a challenge, as they do for all current generative models.
    *   **Logical and Causal Consistency:** Ensuring strict logical consistency over long durations or across multiple complex shots (e.g., a character's shirt getting wet from rain and staying wet in subsequent scenes) is an ongoing research problem.
    *   **Computational Cost:** Despite the efficiency strategy, training and running a model of this scale remains extremely resource-intensive, limiting its accessibility to well-funded research labs and corporations.
    *   **Lack of Ablation Studies:** The paper presents the final, complex system without empirically justifying each design choice through ablation studies, making it harder to discern the individual impact of each architectural innovation.

*   **Potential Future Work:**
    *   **Longer Durations:** Extending the generation capability beyond 15 seconds to create minute-long or even feature-length videos with coherent narratives.
    *   **Interactive Control:** Moving beyond prompt-based generation to more interactive and controllable systems, where users can directly manipulate objects and events in the generated video in real-time.
    *   **Improved World Modeling:** Enhancing the model's internal understanding of physics, causality, and common sense to generate more realistic and logically consistent content.
    *   **Open-Sourcing and Accessibility:** Releasing parts of the model or more detailed architectural information to the broader research community to foster collaboration and innovation.

## 7.3. Personal Insights & Critique
*   **Strengths and Inspirations:**
    *   The **principle of unification** is the most impressive aspect of this work. SkyReels-V4 represents a significant step towards an "all-in-one" creative tool, consolidating what would previously require a chain of multiple specialized models into a single, coherent system. This is a powerful paradigm for both usability and performance, as joint end-to-end training can capture synergies that pipelined approaches miss.
    *   The **channel-concatenation for inpainting** is an elegant and powerful abstraction. It simplifies a diverse set of complex tasks into a single, intuitive mechanism. This idea could be highly influential and applicable to other generative domains beyond video.
    *   The **pragmatic efficiency strategy** (low-res full video + high-res keyframes) is a clever engineering solution to a major bottleneck in high-resolution video generation. It acknowledges the practical constraints of current hardware and provides a viable path to cinematic quality without requiring a theoretical breakthrough in computational efficiency.

*   **Potential Issues and Critique:**
    *   **Reproducibility and Transparency:** As a technical report from a commercial entity without open-sourced code or models, the work is not reproducible by the academic community. The claims, while supported by strong human evaluation results, cannot be independently verified.
    *   **Benchmark Bias:** The primary human evaluation is conducted on a benchmark (SkyReels-VABench) developed by the same team. While the methodology appears sound, there is an inherent risk of the benchmark prompts being unintentionally tailored to the strengths of their own model. The strong performance on the independent Artificial Analysis Arena helps mitigate this concern, but it's still a point to consider.
    *   **The "First" Claim:** The paper claims to be the "first" model to unify all these capabilities. In the rapidly evolving field of generative AI, such claims are hard to definitively prove and can become outdated quickly. While the combination of features is certainly novel and state-of-the-art, the core components (dual-stream transformers, MLLM conditioning, etc.) build upon a rich body of existing research. The innovation lies more in the integration and execution than in the invention of entirely new primitives.

        Overall, SkyReels-V4 is a landmark piece of engineering that showcases a mature, highly capable, and thoughtfully designed system for multi-modal media creation. It effectively synthesizes the latest advances in the field into a unified whole, setting a high bar for the next generation of video foundation models.