# 1. Bibliographic Information

## 1.1. Title
The central topic of this paper is `Pack and Force Your Memory: Long-form and Consistent Video Generation`. It focuses on developing methods to generate extended video sequences while maintaining temporal coherence and mitigating common issues in autoregressive models.

## 1.2. Authors
The authors of this paper are:
*   Xiaofei Wu (ShanghaiTech University, Tencent Hunyuan)
*   Guozhen Zhang (Nanjing University, Tencent Hunyuan)
*   Zhiyong Xu (Tencent Hunyuan)
*   Yuan Zhou (Tencent Hunyuan)
*   Qinglin Lu (Tencent Hunyuan)
*   Xuming He (ShanghaiTech University)

    Their affiliations suggest a collaboration between academic institutions (ShanghaiTech University, Nanjing University) and an industry research lab (Tencent Hunyuan), indicating a blend of theoretical research and practical application expertise in artificial intelligence, particularly in generative models and computer vision.

## 1.3. Journal/Conference
The paper is published as a preprint on arXiv. While `arXiv` itself is not a peer-reviewed journal or conference, it serves as a common platform for researchers to share their work before, or in parallel with, formal publication. Given the affiliations and the nature of the research in advanced video generation, it is likely intended for submission to a top-tier computer vision or machine learning conference (e.g., CVPR, ICCV, NeurIPS, ICLR) or journal. These venues are highly reputable and influential in the field of AI and computer vision.

## 1.4. Publication Year
The paper was published at (UTC): `2025-10-02T08:22:46.000Z`, indicating a publication year of 2025.

## 1.5. Abstract
Long-form video generation faces two main challenges: effectively capturing long-range dependencies across many frames and preventing the accumulation of errors inherent in `autoregressive decoding`. To address these, the paper introduces two primary contributions.

First, for dynamic context modeling, they propose `MemoryPack`, a learnable context-retrieval mechanism. `MemoryPack` uses both textual prompts and image information as global guidance to jointly model both short-term (local motion and appearance) and long-term (global narrative, object identity) dependencies, achieving minute-level temporal consistency. This mechanism is designed to scale efficiently with video length, maintaining linear computational complexity.

Second, to combat error accumulation, they present `Direct Forcing`, an efficient single-step approximation strategy. This strategy improves the alignment between the training process (where models usually see ground-truth data) and the inference process (where models rely on their own predictions), thereby reducing the propagation of errors during long video generation.

Collectively, `MemoryPack` and `Direct Forcing` significantly enhance the contextual consistency and overall reliability of long-form video generation, making autoregressive video models more practically usable.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2510.01784
*   **PDF Link:** https://arxiv.org/pdf/2510.01784v2.pdf

    The publication status is "preprint" on `arXiv`.

# 2. Executive Summary

## 2.1. Background & Motivation

The core problem the paper aims to solve is the effective and efficient generation of `long-form videos` (minute-scale or longer) while maintaining high temporal coherence and mitigating `error accumulation`.

This problem is highly important in the current field due to the growing demand for applications in `content creation` (e.g., film, animation), `embodied intelligence` (e.g., realistic robot simulation), and `interactive gaming` (e.g., dynamic virtual environments). Recent advances in `Diffusion Transformer (DiT)` models have shown strong capabilities in generating realistic `short video clips` (second-level). However, extending these models to `long-form videos` presents significant challenges:
*   **Computational Prohibitions:** The `quadratic complexity` ($\mathcal{O}(L^2)$ where $L$ is token count) of standard `DiT` architectures becomes computationally prohibitive for the substantially larger token counts in long videos.
*   **Lack of Effective Long-Term Modeling:** Existing approaches for long videos often rely on limited context windows or rigid compression strategies, primarily using local visual information. This leads to a degradation in `temporal coherence` and `global consistency` (e.g., object identities, scene layouts, overall narrative) as video length increases, causing `drift` and `flickering`.
*   **Error Accumulation:** `Autoregressive decoding`, where future frames are predicted based on previously generated ones, inevitably leads to `training-inference mismatch`. During training, models condition on `ground-truth frames`, but during inference, they condition on their own potentially error-prone `self-predictions`. These errors compound over long horizons, severely degrading the quality and consistency of generated videos.

    The paper's entry point or innovative idea is to reformulate long video generation as a `long-short term information retrieval problem` and to address the training-inference mismatch with an efficient single-step approximation. This involves creating a dynamic, learnable memory mechanism that leverages multimodal cues (text and image) for global guidance, combined with a novel training strategy to align inference with training effectively.

## 2.2. Main Contributions / Findings

The paper's primary contributions are threefold:

1.  **MemoryPack:** A novel `dynamic memory mechanism` for `dynamic context modeling`.
    *   It jointly models `long-term` and `short-term dependencies` by leveraging textual prompts and an input image as `global guidance`.
    *   It efficiently retrieves historical context, reinforcing `temporal coherence` across minute-level videos.
    *   Unlike rigid compression or fixed frame selection methods, `MemoryPack` enables flexible associations and maintains `linear computational complexity` ($\mathcal{O}(n)$ with respect to video length), ensuring scalability and computational efficiency.
    *   It incorporates `RoPE Consistency` to explicitly encode relative positions across segments, mitigating flickering and discontinuities.

2.  **Direct Forcing:** An `efficient single-step approximating strategy` to mitigate `error accumulation`.
    *   It aligns the training process with model inference in a single step, addressing the `training-inference mismatch`.
    *   Built on `rectified flow`, it performs a one-step backward `ODE` computation in the predicted vector field to approximate inference outputs.
    *   This method incurs `no additional overhead`, `requires no distillation`, preserves `train-inference consistency`, and effectively `curtails error propagation` during inference.

3.  **State-of-the-Art Performance:**
    *   Extensive evaluations demonstrate that the combined approach achieves `state-of-the-art results` on `VBench` across key metrics such as `Motion Smoothness`, `Background Consistency`, and `Subject Consistency`.
    *   The method shows substantially `enhanced robustness against error accumulation` and achieves the best overall performance in `human evaluations` (ELO-K32 scores).
    *   Qualitative results highlight superior `consistency preservation` and `interaction capability` over minute-long sequences, including accurate object reconstruction after prolonged disappearance.

        These findings collectively solve the specific problems of maintaining temporal coherence and mitigating error accumulation in long-form video generation, significantly advancing the practical usability and quality of autoregressive video models.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

To understand this paper, a beginner should be familiar with the following core concepts:

### 3.1.1. Autoregressive Decoding
`Autoregressive decoding` is a common strategy in sequence generation models where each element in a sequence is predicted based on the preceding elements. For video generation, this means generating frame $t$ conditioned on frame `t-1`, frame `t-2`, and so on.
*   **Benefits:** It allows for generation of arbitrarily long sequences and often captures local dependencies well.
*   **Challenge (Error Accumulation):** The main drawback in generative tasks is that if an error is made at an earlier step, it can be propagated and compounded in subsequent steps because the model conditions on its *own generated output* rather than the `ground-truth`. This leads to a `training-inference mismatch`, as models are typically trained by conditioning on `ground-truth` data.

### 3.1.2. Diffusion Models and Rectified Flow
`Diffusion Models` (DMs) are a class of generative models that learn to reverse a gradual diffusion process. They iteratively denoise data from a simple distribution (like Gaussian noise) to generate complex data (like images or videos).
*   **Forward Diffusion Process:** Gradually adds noise to data until it becomes pure noise.
*   **Reverse Diffusion Process:** Learns to reverse this process, starting from noise and iteratively removing it to generate data. This reverse process often involves solving `Stochastic Differential Equations (SDEs)` or `Ordinary Differential Equations (ODEs)`.
*   **Rectified Flow:** A specific type of `ODE-based diffusion model` that aims to learn a straight-line path (rectified flow) between noise and data distributions. This simplifies the reverse process, making it more direct and often allowing for fewer steps (even a single step) to generate high-quality samples. The idea is that instead of a complex, winding path, the model learns a simple, linear trajectory in the latent space.

### 3.1.3. Transformers and Attention Mechanism
`Transformers` are a neural network architecture, originally for natural language processing, that relies heavily on the `attention mechanism`.
*   **Attention Mechanism:** Allows the model to weigh the importance of different parts of the input sequence when processing each element. Instead of processing elements sequentially like `Recurrent Neural Networks (RNNs)`, attention allows parallel processing and direct access to any part of the input.
*   **Self-Attention:** A variant where the attention mechanism processes a single sequence, relating different positions of the sequence to compute a representation of the same sequence. For a given input sequence of tokens, each token interacts with all other tokens to compute its new representation.
    *   It uses three learned linear projections: `Query (Q)`, `Key (K)`, and `Value (V)` matrices.
    *   The `Attention` calculation is:
        \$
        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
        \$
        Where:
        *   $Q$ is the `Query` matrix (from the current token wanting to attend).
        *   $K$ is the `Key` matrix (from all tokens being attended to).
        *   $V$ is the `Value` matrix (from all tokens, providing the information).
        *   $d_k$ is the dimension of the `Key` vectors, used for scaling to prevent very large dot products that push the `softmax` into regions with tiny gradients.
        *   $QK^T$ computes similarity scores between queries and keys.
        *   $\mathrm{softmax}$ normalizes these scores into attention weights.
        *   Multiplying by $V$ produces a weighted sum of `Value` vectors, representing the attended output.
*   **Cross-Attention:** Similar to self-attention, but the `Query` comes from one sequence (e.g., visual features) and `Key` and `Value` come from a different sequence (e.g., text embeddings or memory features). This allows the query sequence to attend to and integrate information from the key/value sequence.
*   **Diffusion Transformer (DiT):** A specific architecture that combines the principles of `Diffusion Models` with the `Transformer` architecture. It replaces the `U-Net` backbone typically used in diffusion models with a `Transformer`, enabling it to model complex spatiotemporal dependencies efficiently. However, standard `DiT` models often have `quadratic computational complexity` ($\mathcal{O}(L^2)$) with respect to the sequence length $L$, which is a major bottleneck for long sequences like videos.

### 3.1.4. Rotary Position Embedding (RoPE)
`Positional embeddings` are crucial in `Transformers` because they are permutation-invariant (order doesn't matter by default). `RoPE` is a method to inject `relative positional information` into self-attention in a way that allows the model to explicitly reason about the relative distances between tokens.
*   Instead of adding a fixed embedding to the input, `RoPE` applies a rotation matrix to the `Query` and `Key` vectors based on their absolute positions. This design ensures that the dot product of $Q$ and $K$ implicitly encodes their relative positions.
*   This is particularly useful for long sequences where knowing the relative distance between tokens is more important than their absolute positions, and it can generalize to longer sequences than seen during training.

## 3.2. Previous Works

The paper discusses several prior studies that highlight the challenges and existing solutions in video generation:

*   **Short-clip Video Generation with DiT:** Models like `DiT` [Peebles and Xie, 2023], `HunyuanVideo` [Kong et al., 2024], `WAN` [Wan et al., 2025], `Seedance 1.0` [Gao et al., 2025], and `Waver` [Zhang et al., 2025b] have demonstrated strong capabilities in generating realistic `fixed-length video clips` (typically second-level). They excel at capturing complex spatiotemporal dependencies and character interactions within these short sequences. The innovation here was moving from `U-Net` based diffusion to `Transformer` based backbones for better scaling and modeling of relationships.

*   **Existing Long-form Video Generation Approaches:**
    *   `Zhang and Agrawala [2025]` (which includes `FramePack-F0` and `FramePack-F1`) and `Teng et al. [2025]` (`Magi-1`) extend `DiT` to generate longer videos by iteratively predicting future frames conditioned on previous ones.
    *   **Limitation:** These methods typically restrict the model to attend only to a `limited window` of recent frames (e.g., `FramePack` uses fixed compression) or apply `fixed compression strategies` to select key frames. While this improves computational efficiency, it inevitably leads to the loss of `long-range information` and degrades `temporal coherence` over extended periods. They primarily rely on local visual cues.
    *   **Mixture of Contexts (MoC) [Cai et al., 2025]:** This approach introduces flexibility by selecting contextual information through routing mechanisms. However, it relies on `manually defined selection rules`, which limits the model's ability to autonomously determine the most relevant information, reducing its adaptability and potentially missing subtle long-range cues.

*   **Error Accumulation Mitigation Strategies:**
    *   `Zhang and Agrawala [2025]`: Attempts to mitigate drift by generating frames in reverse order.
    *   **Limitation:** This strategy, by relying on `tail frames`, might reduce `motion dynamics` in long videos and has limited effectiveness in preventing full error accumulation.
    *   `Huang et al. [2025]` (`Self Forcing`): Addresses the training-inference mismatch by conditioning directly on generated outputs during training.
    *   **Limitation:** To improve efficiency, it requires `distribution distillation`, which introduces extra computation and can degrade generation quality due to the inherent limitations of distillation. Incorporating previously generated results into the training process can also introduce additional noise.
    *   Other approaches for efficiency in long sequences include `linear attention` [Cai et al., 2022, Gu and Dao, 2023b, Yang et al., 2024a, Wang et al., 2020, Peng et al., 2023, Gu and Dao, 2023a], `test-time training` [Dalal et al., 2025], `KV-cache` for rolling updates [Huang et al., 2025], and `sparse attention` [Xi et al., 2025]. These focus on the $\mathcal{O}(L^2)$ complexity but don't directly tackle the memory or error accumulation issues in the same way.

## 3.3. Technological Evolution

Video generation has rapidly evolved:
1.  **Early Generative Models:** Began with `GANs (Generative Adversarial Networks)` and `VAEs (Variational Autoencoders)` producing short, often low-resolution video snippets.
2.  **Transformer-based Models:** The advent of `Transformers` revolutionized sequence modeling, leading to their application in vision. `Vision Transformers (ViTs)` showed how transformers could process images as sequences of patches.
3.  **Diffusion Models:** Emerged as powerful generative models, excelling in image quality and diversity. The combination of `Diffusion Models` with `U-Net` architectures became standard.
4.  **Diffusion Transformers (DiTs):** Replaced the `U-Net` backbone in diffusion models with `Transformers`, further enhancing scalability and the ability to model complex dependencies, particularly for short video clips. This is where `DiT` models like `HunyuanVideo` and `WAN` stand.
5.  **Addressing Long-form Challenges:** The current frontier, where this paper sits, involves extending `DiT` capabilities to `minute-long videos`. This requires overcoming the `quadratic complexity` of `Transformers` and solving the `temporal consistency` and `error accumulation` issues inherent in `autoregressive generation`. Previous attempts used fixed windows or aggressive compression, leading to coherence issues.

    This paper's work fits within the technological timeline as a significant step towards practical `long-form video generation`, building upon `DiT` architectures but introducing novel memory and training strategies to specifically address the unique challenges of extended temporal coherence and stability.

## 3.4. Differentiation Analysis

This paper's approach differentiates from previous works primarily in two areas:

1.  **Dynamic Context Modeling (MemoryPack) vs. Fixed/Limited Context:**
    *   **Prior methods** (`FramePack` in `Zhang and Agrawala [2025]`, `Magi-1` in `Teng et al., [2025]`) typically rely on `fixed-size context windows` or `rigid compression schemes` (e.g., only the most recent frames). While computationally efficient, these approaches inherently lose `long-range dependencies` and often struggle to maintain `global temporal coherence` (object identity, scene layout) over extended periods. `MoC` [Cai et al., 2025] introduced context selection but required `manually defined rules`.
    *   **MemoryPack's Innovation:** It introduces a `learnable, dynamic memory mechanism` (`SemanticPack`) that *combines* visual features with `global textual and image guidance`. This allows for flexible associations between historical information and future frame generation. By leveraging `text prompts` and a `reference image`, it provides a semantic prior that anchors the memory, reinforcing `long-term coherence` in a `semantically grounded` way. The `linear computational complexity` is also a key differentiator against `quadratic complexity` `DiT` models. Furthermore, `RoPE Consistency` explicitly preserves cross-segment positional information, a nuance missed by simple windowing.

2.  **Training-Inference Alignment (Direct Forcing) vs. Multi-step/Distillation/Reverse Generation:**
    *   **Prior methods** often suffer from `error accumulation` due to the `training-inference mismatch`. Approaches like `Zhang and Agrawala [2025]` tried `reverse-order generation`, which can impact `motion dynamics`. `Huang et al. [2025]` (`Self Forcing`) addressed this by conditioning on generated outputs but required `distribution distillation` for efficiency, introducing extra computation and potentially degrading quality.
    *   **Direct Forcing's Innovation:** It proposes an `efficient single-step approximation strategy` based on `rectified flow`. Instead of multi-step inference or distillation, it performs a one-step backward `ODE` computation to approximate inference outputs during training. This `directly aligns training with inference` in a computationally efficient manner, `without additional overhead` or the need for distillation, thus effectively mitigating `error propagation` and preserving generation quality. This direct alignment in a single step is a key improvement over previous `forcing` mechanisms.

        In essence, the paper tackles both the *memory* aspect (what information to retain and how) and the *learning stability* aspect (how to train reliably under autoregressive generation) simultaneously, offering a more holistic solution for robust long-form video generation.

# 4. Methodology

## 4.1. Principles

The core idea of the proposed method is to address the dual challenges of `long-form video generation`: capturing `long-range dependencies` and preventing `error accumulation` in `autoregressive decoding`. The theoretical basis is rooted in enhancing `Diffusion Transformer (DiT)` architectures with a dynamic, multimodal memory mechanism and a rectified-flow-based training strategy that directly aligns training and inference.

The intuition behind `MemoryPack` is that generating long videos requires a balance: very local information for smooth motion (`short-term context`) and a global understanding of the narrative, characters, and scene (`long-term context`). Instead of rigid, fixed-window approaches, `MemoryPack` dynamically retrieves and integrates both types of context, guided by explicit `text prompts` and an `input image`, ensuring semantic grounding and temporal coherence over extended periods. This memory is designed to be computationally efficient, scaling linearly with video length.

The intuition behind `Direct Forcing` is to bridge the `training-inference mismatch` problem. In autoregressive models, training usually conditions on perfect `ground-truth` data, but at inference, the model must rely on its own imperfect `predictions`. This discrepancy causes errors to compound. `Direct Forcing` leverages the concept of `rectified flow`, which provides a simpler, more direct path between noise and data. By approximating the inference output in a single step using this rectified flow, the model can be trained under conditions closer to actual inference, thereby mitigating error accumulation without incurring the high computational cost of multi-step simulations or distillation.

## 4.2. Core Methodology In-depth (Layer by Layer)

The proposed approach formulates video generation as an `autoregressive image-to-video generation task`. Given $n$ historical segments (latent representations of frames) $\{ \mathbf { x } ^ { 0 } , \dotsc , \mathbf { x } ^ { n - 1 } \}$ , a textual prompt $P$, and a conditional image $I$, the objective is to generate the subsequent segment $\mathbf { x } ^ { n }$. The backbone is a `Diffusion Transformer (DiT)` architecture [Peebles and Xie, 2023, Kong et al., 2024], modified for autoregressive generation.

The overall architecture is illustrated in Figure 1, showcasing how the `MemoryPack` module feeds `long-term` and `short-term context` into the `MM-DiT` (Multimodal Diffusion Transformer) backbone, and `Direct Forcing` ensures robust training.

![Figure 1: Overview of our framework. Given a text prompt, an input image, and history frames, the model autoregressively generates future frames. Prior to feeding data into MM-DiT, MemoryPack retrieves both long- and short-term context. In SemanticPack, visual features are extracted within local windows via self-attention, followed by cross-attention to align them with global textual and visual information to iteratively generate long-term dependencies $\\psi _ { n }$ . This design achieves linear computational complexity and substantially improves the efficiency of long-form video generation.](images/1.jpg)
*该图像是示意图，展示了提出的框架的结构。左侧部分展示了MM-DiT模型在生成未来帧过程中如何通过MemoryPack模块获取长期和短期的上下文信息。右侧部分详细介绍了SemanticPack模块的工作机制，包括自注意力和记忆机制的应用，以提取和加强长期依赖特征 $\psi_n$。该设计旨在实现线性计算复杂度，提高长视频生成的效率。*

Figure 1: Overview of our framework. Given a text prompt, an input image, and history frames, the model autoregressively generates future frames. Prior to feeding data into MM-DiT, MemoryPack retrieves both long- and short-term context. In SemanticPack, visual features are extracted within local windows via self-attention, followed by cross-attention to align them with global textual and visual information to iteratively generate long-term dependencies $\psi _ { n }$ . This design achieves linear computational complexity and substantially improves the efficiency of long-form video generation.

### 4.2.1. MemoryPack

`MemoryPack` is a hierarchical module designed to jointly leverage `complementary short-term and long-term contexts` for video generation, balancing local motion fidelity with global semantic coherence. It consists of two components: `FramePack` and `SemanticPack`.

#### 4.2.1.1. FramePack
`FramePack` focuses on `short-term context`. It captures appearance and motion information from recent frames using a fixed compression scheme. This component is effective at enforcing `short-term consistency` (e.g., smooth transitions between adjacent frames). However, its reliance on a fixed window size and compression ratio limits its ability to dynamically propagate information over long time horizons, making it insufficient for `global temporal coherence`. The paper uses `Framepack-F1` as the backbone, suggesting this component is inherited from prior work [Zhang and Agrawala, 2025].

#### 4.2.1.2. SemanticPack
`SemanticPack` is the novel component for maintaining `global temporal coherence`. It integrates visual features with `textual and image guidance`, unlike prior methods that rely solely on visual representations. This is achieved by iteratively updating a `long-term memory representation` $\psi$ using contextual video segments, a text prompt $P$, and a reference image $I$. The process involves two structured operations:

1.  **Memorize:** This operation applies `self-attention` within local windows of historical segments $\{ \mathbf { x } ^ { 0 } , \ldots , \mathbf { x } ^ { n - 1 } \}$ to produce compact embeddings. By processing within windows, it mitigates the prohibitive `quadratic complexity` of attending to the entire long history while still retaining holistic window-level cues. This step effectively compresses the historical visual information into a more manageable form.

2.  **Squeeze:** This operation injects the `textual and image guidance` into the visual memory generated by `Memorize`. Following prior work [Wan et al., 2025], this is implemented as a `cross-attention layer`. The output of `Memorize` serves as the `query (Q)`, and the current `long-term memory representation` $\psi$ acts as the `key (K)` and `value (V)`. This alignment ensures that the `long-term memory` $\psi$ remains `globally aware` and `semantically grounded` by integrating information from the `text prompt` and `reference image`. The iterative update of $\psi$ is formalized as:
    $$
    \psi _ { n + 1 } = \mathrm { Squeeze } \big ( \psi _ { n } , \mathrm { Memorize } ( \mathbf { x } ^ { n } ) \big ) .
    $$
    Where:
    *   $\psi_{n+1}$ is the updated long-term memory representation for the next time step.
    *   $\psi_n$ is the current long-term memory representation.
    *   $\mathrm{Memorize}(\mathbf{x}^n)$ is the compact visual embedding of the current segment $\mathbf{x}^n$, obtained by applying self-attention within its local window.
    *   $\mathrm{Squeeze}(\cdot, \cdot)$ is the cross-attention layer that integrates the visual embedding with the current memory, guided by global multimodal information.

        For initialization, $\psi_0$ is set as the concatenation of the `prompt feature` and the `reference image feature`. This provides a strong `semantic prior` that anchors the memory trajectory from the very beginning. Importantly, the `computational complexity` of `SemanticPack` is $\mathcal{O}(n)$ with respect to the number of historical frames, which ensures `scalability` for long videos by avoiding prohibitive cost growth.

#### 4.2.1.3. RoPE Consistency
In `DiT-based autoregressive video generation`, long videos are typically partitioned into multiple segments during training. This segmentation can lead to a loss of `cross-segment positional information`, causing `flickering` or `temporal discontinuities` even between adjacent segments. To address this:
*   The `input image` is treated as a `CLS-like token` (similar to the class token in Vision Transformers).
*   `Rotary Position Embedding (RoPE)` [Su et al., 2024] is incorporated to explicitly encode `relative positions` across segments.
*   During training, for each video clip, the image is assigned the `initial index` of the entire video. This helps preserve `coherence` and enhances `global temporal consistency` by providing a stable positional anchor.

    Formally, `RoPE` satisfies the property for a given query $\mathbf{x}_q$ at position $m$ and a key $\mathbf{x}_k$ at position $n$:
$$
R _ { q } ( \mathbf { x } _ { q } , m ) R _ { k } ( \mathbf { x } _ { k } , n ) = R _ { g } ( \mathbf { x } _ { q } , \mathbf { x } _ { k } , n - m ) , \quad \Theta _ { k } ( \mathbf { x } _ { k } , n ) - \Theta _ { q } ( \mathbf { x } _ { q } , m ) = \Theta _ { g } ( \mathbf { x } _ { q } , \mathbf { x } _ { k } , n - m ) .
$$
Where:
*   $R_q$ and $R_k$ are the rotation matrices applied to the `query` and `key` vectors, respectively, based on their absolute positions $m$ and $n$.
*   $R_g$ is a rotation matrix that directly depends on the relative position `n-m`. This means the dot product between the rotated $Q$ and $K$ vectors effectively captures their `relative position`.
*   $\Theta_q$ and $\Theta_k$ are the rotation angles for the `query` and `key` vectors.
*   $\Theta_g$ represents the rotation angle corresponding to the relative position difference `n-m`.
*   $\mathbf{x}_q$ and $\mathbf{x}_k$ are the `query` and `key` vectors themselves.

    By assigning the image token a starting index, the sequence can jointly capture both `absolute positions` across video segments and `relative dependencies` within each segment, thereby mitigating flickering and discontinuities.

### 4.2.2. Direct Forcing

`Direct Forcing` addresses `error accumulation` caused by the `training-inference mismatch` in `autoregressive video generation`. During training, the model conditions on `ground-truth frames`, but during inference, it must rely on its own previously `generated outputs`, causing errors to compound. This strategy uses a `rectified-flow-based single-step approximation` to align training and inference trajectories efficiently.

#### 4.2.2.1. Training with Rectified Flow
Following the `rectified flow` formulation [Liu et al., 2022], a linear interpolation is defined between the video distribution $\mathbf{X}$ and Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$.
For simplicity, we omit the superscript of $\mathbf{X}$ and use the subscript to indicate timestep $t$:
$$
\mathbf { x } _ { t } = t \mathbf { x } + ( 1 - t ) { \boldsymbol { \epsilon } } , \quad t \in [ 0 , 1 ] .
$$
Where:
*   $\mathbf{x}_t$ is the interpolated point at time $t$.
*   $\mathbf{x}$ is the target video segment (ground-truth data).
*   $\boldsymbol{\epsilon}$ is a sample from standard Gaussian noise.
*   $t$ is a continuous time variable ranging from 0 to 1. When $t=0$, $\mathbf{x}_0 = \boldsymbol{\epsilon}$ (pure noise); when $t=1$, $\mathbf{x}_1 = \mathbf{x}$ (pure data).

    The `instantaneous velocity` along this trajectory is given by the derivative with respect to $t$:
$$
{ \pmb u } _ { t } = \frac { d { \bf x } _ { t } } { d t } = { \bf x } - \epsilon ,
$$
Where:
*   $\mathbf{u}_t$ is the velocity vector at point $\mathbf{x}_t$, which defines an `Ordinary Differential Equation (ODE)` that guides $\mathbf{x}_t$ toward the target $\mathbf{x}$. The model's task is to predict this velocity field.

    The model predicts a velocity field $v_{\theta}(\mathbf{x}_t, t)$, parameterized by $\theta$. The parameters are optimized by minimizing the `flow matching loss`:
$$
\mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { t , \mathbf { x } , \epsilon } \big [ | | v _ { \theta } ( \mathbf { x } _ { t } , t ) - { \mathbf { u } } _ { t } | | ^ { 2 } \big ] .
$$
Where:
*   $\mathcal{L}_{\mathrm{FM}}(\theta)$ is the flow matching loss function.
*   $\mathbb{E}_{t, \mathbf{x}, \epsilon}$ denotes the expectation over random samples of $t$, ground-truth data $\mathbf{x}$, and noise $\boldsymbol{\epsilon}$.
*   $v_{\theta}(\mathbf{x}_t, t)$ is the velocity field predicted by the model at $\mathbf{x}_t$ and time $t$.
*   $\mathbf{u}_t$ is the true instantaneous velocity.
*   $||\cdot||^2$ is the squared Euclidean norm, ensuring the model's predicted velocity matches the true velocity.

#### 4.2.2.2. Single-Step Approximation
During inference, a video segment $\hat{\mathbf{x}}_1$ is ideally generated by reverse-time integration of the `ODE` starting from $\mathbf{x}_0 \sim \mathcal{N}(0, I)$ to obtain $\hat{\mathbf{x}}_1$:
$$
\hat { \mathbf { x } } _ { 1 } = \int _ { 0 } ^ { 1 } v _ { \theta } ( \mathbf { x } _ { t } , t ) d t .
$$
This integral usually requires `multi-step inference`, which incurs substantial computational costs, especially during training when simulating inference conditions (as in `Student Forcing` [Bengio et al., 2015]).

`Direct Forcing` aligns training with inference by approximating the trajectory in a *single step*, without high computational costs. This is based on the intuition that `rectified flow` guarantees a more direct `ODE trajectory`.
The single-step approximation $\tilde{\mathbf{x}}_1$ of the target $\hat{\mathbf{x}}_1$ is given by:
$$
\tilde { \mathbf { x } } _ { 1 } = \mathbf { x } _ { t } + \Delta _ { t } * v _ { \theta } ( \mathbf { x } _ { t } , t ) \approx \hat { \mathbf { x } } _ { 1 } , \quad \Delta _ { t } = 1 - t .
$$
Where:
*   $\tilde{\mathbf{x}}_1$ is the single-step approximation of the generated data.
*   $\mathbf{x}_t$ is the noisy input at time $t$.
*   $v_{\theta}(\mathbf{x}_t, t)$ is the velocity predicted by the model.
*   $\Delta_t = 1 - t$ is the remaining "time-to-target" step size. Intuitively, this equation takes the current noisy point $\mathbf{x}_t$ and moves it along the predicted velocity vector $v_{\theta}(\mathbf{x}_t, t)$ for the remaining time $\Delta_t$ to directly estimate the final data point $\hat{\mathbf{x}}_1$.

    This process is illustrated in Figure 2, highlighting its efficiency compared to `Student Forcing`.

    ![Figure 2: Schematic illustration of the approximation process. In Student Forcing, multi-step inference is applied to approximate $\\hat { \\bf x } _ { 1 }$ , but this incurs substantial computational overhead and slows training convergence. In contrast, Direct Forcing applies a single-step transformation from $\\mathbf { x } _ { 1 }$ to $\\mathbf { x } _ { t }$ , followed by a denoising step that produces $\\tilde { \\mathbf { x } } _ { 1 }$ as an estimate of $\\hat { \\mathbf { x } } _ { 1 }$ . This approach incurs no additional computational burden, thereby enabling faster training.](images/2.jpg)
    *该图像是示意图，展示了两种推断方法：Student Forcing 和 Direct Forcing。左侧部分描绘了 Student Forcing 的过程，通过多步推断近似 $\hat{x}_1$，并引入逆高斯噪声；而右侧部分则展示了 Direct Forcing，仅需一步转换从 $x_1$ 到 $x_t$，并注入高斯噪声，通过近似得到估计值 $\tilde{x}_1$。该方法相较于 Student Forcing 更加高效，无额外计算负担。*

Figure 2: Schematic illustration of the approximation process. In Student Forcing, multi-step inference is applied to approximate $\hat { \bf x } _ { 1 }$ , but this incurs substantial computational overhead and slows training convergence. In contrast, Direct Forcing applies a single-step transformation from $\mathbf { x } _ { 1 }$ to $\mathbf { x } _ { t }$ , followed by a denoising step that produces $\tilde { \mathbf { x } } _ { 1 }$ as an estimate of $\hat { \mathbf { x } } _ { 1 }$ . This approach incurs no additional computational burden, thereby enabling faster training.

In practice, this means that during training, instead of always using the `ground-truth` $\mathbf{X}^{i-1}$ to condition the generation of $\mathbf{X}^i$, the model first uses the `ground-truth data` $\mathbf{X}^{i-1}$ (or a noisy version of it) and Eq. 7 to obtain a `one-step approximation` $\tilde{\mathbf{x}}^{i-1}$. This approximation then serves as the `conditional input` for generating $\mathbf{x}^i$ during training. This strategy exposes the model to `inference-like conditions` (where it sees slightly imperfect, model-predicted inputs) and effectively mitigates the `distribution mismatch` while reducing `error accumulation`.

#### 4.2.2.3. Optimization Strategy
To reinforce `temporal continuity` across segments, `Direct Forcing` samples clips from the same video in chronological order and uses them as conditional inputs for iterative training. However, applying a full `backpropagation` and `optimizer update` at every step can perturb the learned distribution and impair consistency.
To address this, the authors adopt `gradient accumulation`: gradients are aggregated over multiple clips before performing a single parameter update. This strategy stabilizes optimization and improves `cross-clip consistency` and `long-range temporal coherence` in generated videos.

# 5. Experimental Setup

## 5.1. Datasets

The training dataset is primarily sourced from:
*   **Mira [Ju et al., 2024]:** A large-scale video dataset with long durations and structured captions.
*   **Sekai [Li et al., 2025b]:** A video dataset oriented towards world exploration.

    These datasets comprise approximately `16,000 video clips` with a total duration of `150 hours` across diverse scenarios. The longest videos in both datasets extend up to `one minute`. To ensure data quality, `dynamism and shot-cut filtering` are applied to all samples. These datasets were chosen because they provide a large volume of video data, crucial for training robust generative models, and include `long-duration videos` up to one minute, which are essential for validating `long-form video generation` capabilities.

## 5.2. Evaluation Metrics

The generated videos are assessed using a comprehensive set of quantitative metrics, categorized by their focus, along with qualitative human evaluations.

### 5.2.1. Imaging Quality
*   **Conceptual Definition:** This metric quantifies distortions in generated frames, such as over-exposure, noise, and blur. It assesses the perceptual fidelity and visual clarity of individual frames.
*   **Mathematical Formula:** The paper uses `MUSIQ [Ke et al., 2021]` trained on the `SPAQ [Fang et al., 2020]` dataset. `MUSIQ` (Multi-scale Image Quality Transformer) is a learned image quality predictor. It doesn't have a simple, single mathematical formula, but rather is a complex neural network model that outputs a quality score. The score typically ranges from 0 to 100, where higher is better.
*   **Symbol Explanation:** Not applicable for a neural network predictor; the output is a scalar quality score.

### 5.2.2. Aesthetic Quality
*   **Conceptual Definition:** This metric evaluates the overall aesthetic appeal of each video frame, considering factors like composition, color richness and harmony, photorealism, naturalness, and artistic value.
*   **Mathematical Formula:** The paper employs the `LAION aesthetic predictor [Schuhmann et al., 2022]`. Similar to `MUSIQ`, this is a neural network model (often a `CLIP`-based model finetuned on aesthetic ratings) that outputs an aesthetic score, typically ranging from 1 to 10, where higher is better.
*   **Symbol Explanation:** Not applicable for a neural network predictor; the output is a scalar aesthetic score.

### 5.2.3. Dynamic Degree
*   **Conceptual Definition:** This metric estimates the amount and intensity of motion present in the synthesized videos, indicating how dynamic or static the generated content is.
*   **Mathematical Formula:** The paper uses `RAFT [Teed and Deng, 2020]` (Recurrent All-Pairs Field Transforms for Optical Flow) to estimate motion. The dynamic degree is typically computed as the magnitude of the `optical flow` vectors.
    Let $F_t$ be the optical flow field between frame $t$ and frame $t+1$. The magnitude of the flow vector at pixel `(x,y)` is $||\mathbf{v}_{x,y}|| = \sqrt{u_{x,y}^2 + v_{x,y}^2}$, where $u_{x,y}$ and $v_{x,y}$ are the horizontal and vertical components of the flow. The `Dynamic Degree` for a video $V$ could be defined as:
    \$
    \mathrm{DynamicDegree}(V) = \frac{1}{T-1} \sum_{t=1}^{T-1} \left( \frac{1}{H \times W} \sum_{x=1}^W \sum_{y=1}^H ||\mathbf{v}_{x,y,t}|| \right)
    \$
*   **Symbol Explanation:**
    *   $V$: The generated video.
    *   $T$: Total number of frames in the video.
    *   `H, W`: Height and width of the video frames.
    *   $\mathbf{v}_{x,y,t}$: Optical flow vector at pixel `(x,y)` between frame $t$ and frame $t+1$.
    *   $||\cdot||$: Euclidean norm (magnitude of the vector).

### 5.2.4. Motion Smoothness
*   **Conceptual Definition:** This metric evaluates the fluidity and naturalness of motion transitions between frames, aiming to detect jerky or unnatural movements.
*   **Mathematical Formula:** The paper leverages motion priors from a `video frame interpolation model [Li et al., 2023]`, as adapted by `VBench`. This typically involves assessing the consistency or predictability of motion vectors over time. A common approach involves calculating the difference in motion vectors between consecutive frames or using a trained model to predict a smoothness score. If based on optical flow, it could involve a metric like the variance of optical flow magnitudes or directions over time. Without the exact `VBench` formula, a generalized approach might be:
    \$
    \mathrm{MotionSmoothness}(V) = \frac{1}{T-2} \sum_{t=1}^{T-2} \exp\left(-\alpha \cdot \mathrm{diff}(\mathbf{F}_t, \mathbf{F}_{t+1})\right)
    \$
    where $\mathrm{diff}(\mathbf{F}_t, \mathbf{F}_{t+1})$ could be the average magnitude of the difference between optical flow fields $\mathbf{F}_t$ (from frame $t$ to $t+1$) and $\mathbf{F}_{t+1}$ (from frame $t+1$ to $t+2$). Higher values indicate smoother motion.
*   **Symbol Explanation:**
    *   $V$: The generated video.
    *   $T$: Total number of frames.
    *   $\mathbf{F}_t$: Optical flow field between frame $t$ and $t+1$.
    *   $\alpha$: A scaling constant.
    *   $\mathrm{diff}(\cdot, \cdot)$: A function to compute the difference between two flow fields, e.g., mean squared difference of flow vectors.

### 5.2.5. Background Consistency
*   **Conceptual Definition:** This metric measures how stable and consistent the background scene remains throughout the video, aiming to detect flickering, changes in layout, or distortions in static elements.
*   **Mathematical Formula:** The paper computes `CLIP [Radford et al., 2021]` feature similarity across frames. `CLIP` (Contrastive Language-Image Pre-training) embeds images into a multimodal latent space. Consistency is then measured by the cosine similarity of these embeddings.
    Let $f_{CLIP}(F_t)$ be the CLIP image embedding for frame $t$.
    \$
    \mathrm{BackgroundConsistency}(V) = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathrm{cosine\_similarity}(f_{CLIP}(F_t), f_{CLIP}(F_{t+1}))
    \$
    More sophisticated methods may focus on background regions specifically, but the paper implies overall frame similarity for background.
*   **Symbol Explanation:**
    *   $V$: The generated video.
    *   $T$: Total number of frames.
    *   $F_t$: Frame at time $t$.
    *   $f_{CLIP}(F_t)$: `CLIP` image embedding of frame $F_t$.
    *   $\mathrm{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$: Cosine similarity between two vectors.

### 5.2.6. Subject Consistency
*   **Conceptual Definition:** This metric assesses the consistency of a subject's appearance, identity, and structure throughout the video sequence, detecting changes in facial features, clothing, or object form.
*   **Mathematical Formula:** The paper computes `DINO [Caron et al., 2021]` feature similarity between frames. `DINO` (self-supervised vision transformer) provides strong semantic features, often useful for object-level understanding. Similar to CLIP, cosine similarity of these embeddings is used.
    Let $f_{DINO}(F_t)$ be the DINO image embedding for frame $t$.
    \$
    \mathrm{SubjectConsistency}(V) = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathrm{cosine\_similarity}(f_{DINO}(F_t), f_{DINO}(F_{t+1}))
    \$
    Again, this typically implies overall frame similarity or potentially masked regions if subject tracking is employed.
*   **Symbol Explanation:**
    *   $V$: The generated video.
    *   $T$: Total number of frames.
    *   $F_t$: Frame at time $t$.
    *   $f_{DINO}(F_t)$: `DINO` image embedding of frame $F_t$.
    *   $\mathrm{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$: Cosine similarity between two vectors.

### 5.2.7. Exposure Bias Metric ($\Delta_{\mathrm{drift}}^M$)
*   **Conceptual Definition:** This metric evaluates the long-horizon video generation capability by quantifying the change or "drift" in a specific metric $M$ (e.g., imaging quality, aesthetic quality) between the start and end of a generated video. A smaller $\Delta_{\mathrm{drift}}^M$ indicates less degradation over time.
*   **Mathematical Formula:**
    $$
    \Delta _ { \mathrm { d r i f t } } ^ { M } ( V ) = \big | M ( V _ { \mathrm { s t a r t } } ) - M ( V _ { \mathrm { e n d } } ) \big | ,
    $$
*   **Symbol Explanation:**
    *   $V$: The tested video generated by the model.
    *   $M$: Any chosen metric (e.g., `Imaging Quality`, `Aesthetic Quality`, `Background Consistency`, `Subject Consistency`).
    *   $V_{\mathrm{start}}$: Represents the first $15\%$ of frames of the video $V$.
    *   $V_{\mathrm{end}}$: Represents the last $15\%$ of frames of the video $V$. Crucially, the paper states that $V_{\mathrm{end}}$ is taken from the *model's own generated last frames* rather than the final frames of a user-obtained video, to focus on the model's inherent capabilities.
    *   $|\cdot|$: Absolute value, measuring the magnitude of the difference.

### 5.2.8. Human Assessment
*   **Conceptual Definition:** This involves collecting subjective human preferences through `A/B testing`. Evaluators compare pairs of videos (e.g., from different models) and indicate their preference. The results are then aggregated and reported using `ELO-K32 scores` and corresponding rankings. `ELO` is a rating system commonly used in competitive games to rank players, adapted here to rank models based on human preferences. A higher ELO score indicates better perceived quality.
*   **Mathematical Formula:** The `ELO rating system` is a complex algorithm that updates scores based on the outcome of pairwise comparisons. It doesn't have a single simple formula for the final score, but rather an update rule after each comparison. For example, if player A with rating $R_A$ plays player B with rating $R_B$ and has an expected score $E_A = 1 / (1 + 10^{(R_B - R_A)/400})$, if player A wins (actual score $S_A=1$), their new rating $R_A'$ is $R_A + K(S_A - E_A)$, where $K$ is the `K-factor` (e.g., 32).
*   **Symbol Explanation:**
    *   `A/B testing`: A method of comparing two versions (A and B) of a single variable to determine which performs better.
    *   `ELO-K32 scores`: `ELO` rating system scores, where `K32` refers to a `K-factor` of 32, a parameter determining how much scores change after each game. Higher scores are better.
    *   `Rankings`: The ordinal position of each model based on its `ELO score`.

## 5.3. Baselines

The paper compares its method against the following baseline models:
*   `Magi-1 [Teng et al., 2025]`: An `autoregressive video generation` model that attempts to scale up generation. It's a relevant baseline for its focus on longer videos.
*   `FramePack-F0 [Zhang and Agrawala, 2025]`: A variant of `FramePack` that emphasizes visual fidelity but tends to have reduced inter-frame dynamics due to its anti-drift sampling strategy.
*   `FramePack-F1 [Zhang and Agrawala, 2025]`: Another variant of `FramePack`, representing a strong baseline for `image-to-video generation` that considers context packing.

    These baselines are representative because they are recent, state-of-the-art `autoregressive video generation` models that attempt to tackle the problem of `long-form video generation`, making them direct competitors in terms of both methodology and objectives.

## 5.4. Implementation Details

*   **Backbone Model:** The `Framepack-F1` model is adopted as the `backbone` for the `image-to-video generation task`.
*   **Training Infrastructure:** Training is conducted in parallel on `GPU clusters` (each with 96GB memory).
*   **Batch Size:** A batch size of `1` is used.
*   **Training Duration:** Approximately `five days`.
*   **Optimizer:** `AdamW` optimizer is employed.
*   **Initial Learning Rate:** $10^{-5}$.
*   **Training Procedure (Two Stages):**
    1.  **Stage 1 (Teacher Forcing):** The entire network is trained using `teacher forcing`. This means that during training, the model is always conditioned on `ground-truth frames` from the previous step. This stage accelerates convergence and mitigates instability that can arise from `sampling bias` (where the model sees its own potentially erroneous outputs early on).
    2.  **Stage 2 (Direct Forcing Fine-tuning):** Only the `output layer` of the model is fine-tuned using `Direct Forcing`. This stage stabilizes the `backbone` (which was pre-trained in Stage 1) and specifically aligns the training process with inference, substantially reducing `error accumulation` in `autoregressive generation`. This strategic fine-tuning focuses the `Direct Forcing` mechanism on correcting the training-inference mismatch without destabilizing the core generative capabilities learned in Stage 1.

        The evaluation covers three duration ranges: short (10 seconds), medium-length (30 seconds), and long (1 minute). All generated videos are at `480p` resolution and `24 fps`. `VBench` [Huang et al., 2024b] is used as the source for all images, and prompts are rewritten using `Qwen2.5-V1` [Bai et al., 2025].

# 6. Results & Analysis

## 6.1. Core Results Analysis

The experiments compare the proposed method with `FramePack-F0`, `FramePack-F1`, and `Magi-1` on 160 videos (60 of 10s, 60 of 30s, 40 of 60s). The `quantitative results` are presented in Table 1 and Table 2, and `qualitative comparisons` are shown in Figure 3, Figure 4, and Figure 5.

### 6.1.1. Quantitative Results Overview

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="6">Global Metrics</th>
<th colspan="4">Error Accumulation</th>
<th colspan="2">Human Evaluation</th>
</tr>
<tr>
<th>Imaging Quality ↑</th>
<th>Aesthetic Quality ↑</th>
<th>Dynamic Degree ↑</th>
<th>Motion Smoothness ↑</th>
<th>Background Consistency ↑</th>
<th>Subject Consistency ↑</th>
<th>ΔImaging Quality ↓</th>
<th>ΔAesthetic Quality ↓</th>
<th>ΔBackground Consistency ↓</th>
<th>ΔSubject Consistency ↓</th>
<th>ELO ↑</th>
<th>Rank ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Magi-1</td>
<td>54.64%</td>
<td>53.98%</td>
<td>67.5%</td>
<td>99.17%</td>
<td>89.30%</td>
<td>82.33%</td>
<td>4.19%</td>
<td>5.02%</td>
<td>1.53%</td>
<td>0.97%</td>
<td>1434</td>
<td>4</td>
</tr>
<tr>
<td>FramePack-F0</td>
<td>68.06%</td>
<td>62.89%</td>
<td>15.38%</td>
<td>99.27%</td>
<td>92.22%</td>
<td>90.03%</td>
<td>2.34%</td>
<td>2.45%</td>
<td>2.68%</td>
<td>2.99%</td>
<td>1459</td>
<td>3</td>
</tr>
<tr>
<td>FramePack-F1</td>
<td>67.06%</td>
<td>59.11%</td>
<td>53.85%</td>
<td>99.12%</td>
<td>90.21%</td>
<td>83.48%</td>
<td>2.71%</td>
<td>4.57%</td>
<td>1.59%</td>
<td>1.08%</td>
<td>1537</td>
<td>2</td>
</tr>
<tr>
<td>Ours</td>
<td>67.55%</td>
<td>59.75%</td>
<td>48.37%</td>
<td>99.31%</td>
<td>93.25%</td>
<td>91.16%</td>
<td>2.51%</td>
<td>3.25%</td>
<td>1.21%</td>
<td>0.76%</td>
<td>1568</td>
<td>1</td>
</tr>
</tbody>
</table>

**Analysis of Table 1:**

*   **Global Metrics:**
    *   Our method (`Ours`) achieves the **best performance** in `Background Consistency` (93.25%), `Subject Consistency` (91.16%), and `Motion Smoothness` (99.31%). This strongly validates the effectiveness of `MemoryPack` in preserving `long-term temporal coherence` and generating fluid movements, and `Direct Forcing` in stabilizing long-horizon generation.
    *   `FramePack-F0` shows the highest `Imaging Quality` (68.06%) and `Aesthetic Quality` (62.89%). However, it has a significantly lower `Dynamic Degree` (15.38%), which the authors attribute to its anti-drift sampling strategy that prioritizes visual fidelity over motion dynamics. This trade-off is evident.
    *   `Magi-1` produces the highest `Dynamic Degree` (67.5%) but suffers from lower `Imaging Quality`, `Aesthetic Quality`, and noticeably worse `Background Consistency` and `Subject Consistency`, indicating a struggle with maintaining overall coherence despite generating more motion.
*   **Error Accumulation Metrics:**
    *   Our method achieves the **lowest error accumulation** across all metrics: $ΔImaging Quality$ (2.51%), $ΔAesthetic Quality$ (3.25%), $ΔBackground Consistency$ (1.21%), and $ΔSubject Consistency$ (0.76%). This is a critical finding, demonstrating the stability and effectiveness of `Direct Forcing` in maintaining quality over `long-term video generation`. The lower values mean less degradation from the start to the end of the video.
*   **Human Evaluation:**
    *   Our method achieves the highest `ELO score` (1568) and the `Rank 1`, indicating superior overall quality and preference by human evaluators. This provides strong subjective validation, complementing the quantitative metrics.

### 6.1.2. Qualitative Results

*   **30-second videos (Figure 3):** Our method demonstrates fewer `temporal identity shifts` and `geometric distortions` compared to `FramePack-F1` and `Magi-1`. `FramePack-F0` preserves visual fidelity but at the cost of `reduced inter-frame dynamics`, confirming the quantitative trade-offs.

    ![Figure 3: Visualization of 30-second videos comparing all methods in terms of consistency preservation and interaction capability. Prompt: Close-up view of vegetables being added into a large silver pot of simmering broth, with leafy greens and stems swirling vividly in the bubbling liquid. Rising steam conveys warmth and motion, while blurred kitchen elements and natural light in the background create a homely yet dynamic culinary atmosphere.](images/3.jpg)
    *该图像是插图，展示了比较不同方法生成30秒视频的效果。每列展示不同时间点的画面，清晰体现了各种方法在蔬菜加入锅中时的流动性和互动性，包括真实图像及三种生成方法（f0、f1及ours）的表现。*

Figure 3: Visualization of 30-second videos comparing all methods in terms of consistency preservation and interaction capability. Prompt: Close-up view of vegetables being added into a large silver pot of simmering broth, with leafy greens and stems swirling vividly in the bubbling liquid. Rising steam conveys warmth and motion, while blurred kitchen elements and natural light in the background create a homely yet dynamic culinary atmosphere.

*   **60-second videos and error accumulation (Figure 4):** As video length increases, competing methods (`Magi-1`, `FramePack-F0`, `FramePack-F1`) show `more severe error accumulation`, leading to degraded image quality and consistency. Our approach, however, maintains `image quality comparable to the first frame` even over `minute-long sequences`, highlighting the effectiveness of `Direct Forcing`.

    ![Figure 4: Visualization of a 60-second video illustrating the accumulation of errors. Our method maintains image quality comparable to the first frame even over minute-long sequences. Prompt: The sun sets over a serene lake nestled within majestic mountains, casting a warm, golden glow that softens at the horizon. The sky is a vibrant canvas of orange, pink, and purple, with wispy clouds catching the last light. Calm and reflective, the lake's surface mirrors the breathtaking colors of the sky in a symphony of light and shadow. In the foreground, lush greenery and rugged rocks frame the tranquil scene, adding a sense of life and stillness. Majestic, misty mountains rise in the background, creating an overall atmosphere of profound peace and tranquility.](images/4.jpg)
    *该图像是展示了一个60秒的视频生成质量的示意图，展示了在不同时间点（0s, 15s, 30s, 45s和60s）中，采用不同方法生成图像的效果。我们的算法维持了与首帧相当的图像质量，即使在长达一分钟的序列中，也有效控制了错误积累。*

Figure 4: Visualization of a 60-second video illustrating the accumulation of errors. Our method maintains image quality comparable to the first frame even over minute-long sequences. Prompt: The sun sets over a serene lake nestled within majestic mountains, casting a warm, golden glow that softens at the horizon. The sky is a vibrant canvas of orange, pink, and purple, with wispy clouds catching the last light. Calm and reflective, the lake's surface mirrors the breathtaking colors of the sky in a symphony of light and shadow. In the foreground, lush greenery and rugged rocks frame the tranquil scene, adding a sense of life and stillness. Majestic, misty mountains rise in the background, creating an overall atmosphere of profound peace and tranquility.

*   **Consistency visualization after occlusion (Figure 5):** Our method accurately reconstructs objects that remain absent for `extended periods` (e.g., a crab disappearing into its burrow). Even with `heavy occlusion`, the model reconstructs and generates objects with `consistent identity and 2D structure` after long intervals. This validates that `MemoryPack` effectively preserves `long-term contextual information` and enables stable memory for extended video generation. `F0` and `F1` fail to follow the prompt and show error accumulation, while `Magi-1` follows the prompt but fails temporal consistency.

    ![Figure 5: Consistency evaluation on a 60-second video shows that when an object ID is heavily occluded for an extended period, reconstruction remains challenging. Both F0 and F1 fail to follow the prompt and exhibit noticeable error accumulation. Although MAGI-1 follows the prompt, it is unable to maintain temporal consistency. Prompt: On the peaceful, sun-drenched sandy beach, a small crab first retreats into its burrow before reemerging. The lens captures its shimmering shell and discreet stride under the low sun angle. As it slowly crawls outward, the crab leaves a faint trail behind, while its elongated shadow adds a cinematic texture to this tranquil scene.](images/5.jpg)
    *该图像是一个视频生成结果的比较图，展示了在60秒视频中不同方法（mage、f0、f1及ours）对同一场景的处理效果。场景描述为一只小螃蟹在阳光明媚的沙滩上先退入洞中再重新出现。尽管f0和f1未能保持一致性，ours方法显示了更好的时间一致性。*

Figure 5: Consistency evaluation on a 60-second video shows that when an object ID is heavily occluded for an extended period, reconstruction remains challenging. Both F0 and F1 fail to follow the prompt and exhibit noticeable error accumulation. Although MAGI-1 follows the prompt, it is unable to maintain temporal consistency. Prompt: On the peaceful, sun-drenched sandy beach, a small crab first retreats into its burrow before reemerging. The lens captures its shimmering shell and discreet stride under the low sun angle. As it slowly crawls outward, the crab leaves a faint trail behind, while its elongated shadow adds a cinematic texture to this tranquil scene.

## 6.2. Ablation Studies / Parameter Analysis

Ablation studies were conducted using a subset of the training data to conserve computational resources.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="6">Global Metrics</th>
<th colspan="4">Error Accumulation</th>
</tr>
<tr>
<th>Imaging Quality ↑</th>
<th>Aesthetic Quality ↑</th>
<th>Dynamic Degree ↑</th>
<th>Motion Smoothness ↑</th>
<th>Background Consistency ↑</th>
<th>Subject Consistency ↑</th>
<th>ΔImaging Quality ↓</th>
<th>ΔAesthetic Quality ↓</th>
<th>ΔBackground Consistency ↓</th>
<th>ΔSubject Consistency ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>F1 w/CT</td>
<td>54.33%</td>
<td>53.07%</td>
<td>56.77%</td>
<td>98.81%</td>
<td>85.32%</td>
<td>85.64%</td>
<td>4.32%</td>
<td>7.18%</td>
<td>3.22%</td>
<td>1.32%</td>
</tr>
<tr>
<td>MemoryPack</td>
<td>55.31%</td>
<td>60.69%</td>
<td>51.13%</td>
<td>98.86%</td>
<td>88.21%</td>
<td>86.77%</td>
<td>2.47%</td>
<td>4.99%</td>
<td>2.31%</td>
<td>1.88%</td>
</tr>
<tr>
<td>zero-MemoryPack</td>
<td>57.37%</td>
<td>51.91%</td>
<td>62.71%</td>
<td>98.85%</td>
<td>87.31%</td>
<td>86.21%</td>
<td>8.32%</td>
<td>6.31%</td>
<td>2.43%</td>
<td>3.10%</td>
</tr>
<tr>
<td>Student forcing</td>
<td>49.32%</td>
<td>52.33%</td>
<td>59.88%</td>
<td>98.96%</td>
<td>85.32%</td>
<td>82.41%</td>
<td>3.17%</td>
<td>5.11%</td>
<td>2.02%</td>
<td>1.12%</td>
</tr>
<tr>
<td>All</td>
<td>57.65%</td>
<td>55.75%</td>
<td>55.37%</td>
<td>99.11%</td>
<td>87.17%</td>
<td>88.77%</td>
<td>3.21%</td>
<td>4.77%</td>
<td>2.02%</td>
<td>0.99%</td>
</tr>
</tbody>
</table>

**Analysis of Table 2:**

*   **F1 w/CT (FramePack-F1 with Curriculum Training):** This row represents a baseline where `FramePack-F1` is fine-tuned on the authors' dataset. It slightly improved `Dynamic Degree` (56.77%) compared to the original `FramePack-F1` in Table 1 (53.85%), but degraded performance on other metrics (e.g., lower `Imaging Quality` and `Aesthetic Quality`, higher $ΔAesthetic Quality$). This suggests that simply using the dataset doesn't fully unlock the potential, highlighting the need for the proposed `MemoryPack` and `Direct Forcing`.

*   **MemoryPack (without Direct Forcing):** When `MemoryPack` is included but `Direct Forcing` is not, there's an improvement in `Aesthetic Quality` (60.69%), `Background Consistency` (88.21%), and `Subject Consistency` (86.77%) compared to $F1 w/CT$. It also significantly reduces $ΔImaging Quality$ (2.47%) and $ΔAesthetic Quality$ (4.99%), showing that `MemoryPack` alone helps with long-term consistency and reduces error accumulation.

*   **zero-MemoryPack:** This variant tests the `semantic contribution` of `MemoryPack` by initializing the `global memory` $\psi_0$ with a `zero vector` (i.e., no initial semantic guidance from text/image). This led to worse performance on `error-accumulation metrics` (e.g., $ΔImaging Quality$ shot up to 8.32%, $ΔSubject Consistency$ to 3.10%). This confirms that `semantic guidance` from text and image is crucial for stabilizing `long-term video generation` and maintaining consistency.

*   **Student forcing:** This ablation trains the model with its actual sampling process as input (multi-step inference, sampling step set to 5 for efficiency), simulating the `training-inference mismatch`. The resulting performance was `substantially inferior` to `Direct Forcing` (compare `Student forcing` row with `All` row which includes `Direct Forcing`). For example, `Imaging Quality` (49.32%) and `Aesthetic Quality` (52.33%) were lower than `MemoryPack` or `All`. This strongly underscores the effectiveness of `Direct Forcing` in mitigating `error accumulation` and improving overall generation quality for `long-term video generation`.

*   **All (MemoryPack + Direct Forcing):** This row represents the full proposed method (with MemoryPack and Direct Forcing). It achieves the best balance and overall strong performance, especially in `Subject Consistency` (88.77%) and reducing $ΔSubject Consistency$ (0.99%). This confirms the synergistic effect of combining both proposed mechanisms.

### 6.2.1. Validation of SemanticPack Models (Appendix A.1)

The authors conducted further ablation studies on the `Squeeze` operation within `SemanticPack` to evaluate the network's capacity for capturing `long-term dependencies`. They designed three fusion schemes, illustrated in Figure 6.

![Figure 6: Illustration of the optional architecture of SemanticPack.](images/6.jpg)
*该图像是示意图，展示了三种不同的结构替代方案A、B和C，分别用于MemoryPack的实现。图中包括了Squeeze和Memorize模块的相互连接，以及输入和输出节点的关系，清晰地展示了不同结构在信息传递中的作用。*

Figure 6: Illustration of the optional architecture of SemanticPack.

*   **Structure A:** Text and image features are used as `Key (K)` and `Value (V)`, while the visual representation (from `Memorize`) serves as the `Query (Q)`. This is the proposed approach in the main paper.
*   **Structure B:** Text and image features are used as the `Query (Q)`, with the visual representation as `Key (K)` and `Value (V)`.
*   **Structure C:** Concatenates text and image features with the visual representation from the `first window` to enrich the `Query (Q)`, then follows the same setting as (B) for subsequent steps.

    The following are the results from Table 3 of the original paper:

    <table>
    <thead>
    <tr>
    <th rowspan="2">Method</th>
    <th colspan="6">Global Metrics</th>
    </tr>
    <tr>
    <th>Imaging Quality ↑</th>
    <th>Aesthetic Quality ↑</th>
    <th>Dynamic Degree ↑</th>
    <th>Motion Smoothness ↑</th>
    <th>Background Consistency ↑</th>
    <th>Subject Consistency ↑</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>C</td>
    <td>48.31%</td>
    <td>53.15%</td>
    <td>28.98%</td>
    <td>97.72%</td>
    <td>83.27%</td>
    <td>83.74%</td>
    </tr>
    <tr>
    <td>B</td>
    <td>50.11%</td>
    <td>50.71%</td>
    <td>32.78%</td>
    <td>98.91%</td>
    <td>87.11%</td>
    <td>80.56%</td>
    </tr>
    <tr>
    <td>A</td>
    <td>55.31%</td>
    <td>60.69%</td>
    <td>51.13%</td>
    <td>98.86%</td>
    <td>88.21%</td>
    <td>86.77%</td>
    </tr>
    </tbody>
    </table>

**Analysis of Table 3 and Figure 7:**

*   `Structure A` (the proposed method) achieves the best overall performance across metrics like `Imaging Quality`, `Aesthetic Quality`, `Dynamic Degree`, `Background Consistency`, and `Subject Consistency`.
*   `Structures B` and $C$ cause notable degradation in `visual quality` and a substantial reduction in `temporal dynamics`. This is attributed to:
    *   **Structure B:** The limited number of text and image tokens used as `Query` is insufficient to capture adequate visual representations, impairing the model's ability to model dynamics.
    *   **Structure C:** While incorporating initial visual windows into the `Query` increases token count, it introduces `multi-modal information` into the `Query`, which increases training difficulty.
*   Qualitative visualizations (Figure 7) further confirm these findings, showing that `Structures B` and $C$ exhibit `pronounced degradation in temporal dynamics`, `blurred object boundaries`, and `substantially higher exposure bias` compared to `Structure A`. This confirms that feeding multimodal guidance into `Key/Value` and letting visual features query it is the most effective fusion scheme.

    ![Figure 7: Prompt: In a field of golden marigolds, a man and woman stood entwined, their faces glowing with intricate sugar skull makeup beneath the setting sun. The woman, crowned with fiery orange blossoms, gazed at him with tender devotion. He met her eyes, the bold black-and-white patterns on his face striking against his chestnut jacket, his hands gently interlaced with hers. Turning briefly toward the camera, he then lowered his head to kiss her. Behind them, hiltops crowned with golden-domed buildings shimmered beneath a sky of soft blues and pinks, completing the serene, magical scene.](images/7.jpg)
    *该图像是一个视频生成过程的示例，展示了在不同时间点上人物和背景的动态变化。画面中的人物身穿多彩服饰，周围是金色的万寿菊，背景则是由辉煌建筑和柔和天空构成的迷人画面。该图像清晰地体现了长时间的视频生成能力，通过不同的时间帧展示了逐渐变化的场景特征。*

Figure 7: Prompt: In a field of golden marigolds, a man and woman stood entwined, their faces glowing with intricate sugar skull makeup beneath the setting sun. The woman, crowned with fiery orange blossoms, gazed at him with tender devotion. He met her eyes, the bold black-and-white patterns on his face striking against his chestnut jacket, his hands gently interlaced with hers. Turning briefly toward the camera, he then lowered his head to kiss her. Behind them, hiltops crowned with golden-domed buildings shimmered beneath a sky of soft blues and pinks, completing the serene, magical scene.

### 6.2.2. Consistency Visualization (Appendix A.3)

Additional visualizations (Figure 8 and Figure 9) further support the model's ability to maintain `long-term coherence`.
*   **Figure 8:** Shows the model maintaining consistency across 5-second and 30-second sequences, successfully recovering `spatial layouts` despite complex camera motions (panning, zooming).
*   **Figure 9:** Provides minute-level visualizations, highlighting the model's capacity to preserve `scene structure`, maintain `subject integrity`, and avoid `temporal drift` over extended durations.

    ![Figure 8: Visualization of long-term consistency. The model is evaluated on videos of $5 \\mathrm { ~ s ~ }$ and $3 0 \\mathrm { s }$ to assess its ability to maintain coherence.](images/8.jpg)
    *该图像是一个示意图，展示了不同时间点（0s、1s、2s、3s、4s、5s、6s、12s、18s、24s、30s）下生成的视频帧，用于评估模型在长时间序列中的一致性。每行分别展示了不同场景下的视频演变，强调模型在长时间生成中的连贯性。*

Figure 8: Visualization of long-term consistency. The model is evaluated on videos of $5 \mathrm { ~ s ~ }$ and $3 0 \mathrm { s }$ to assess its ability to maintain coherence.

![Figure 9: Visualization of 60s videos.](images/9.jpg)
*该图像是一个展示60秒视频生成的可视化效果，分为0秒、15秒、30秒、45秒和60秒五个阶段。每个阶段展现了不同的场景与变化，从城市夜景到人物动态，表现出随着时间推移的丰富细节与色彩。*

Figure 9: Visualization of 60s videos.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary

This paper introduces a novel framework for `long-form and consistent video generation` by addressing two critical challenges: modeling `long-range dependencies` and mitigating `error accumulation` in `autoregressive decoding`. The core contributions are:

1.  **MemoryPack:** A lightweight and efficient dynamic memory mechanism that models both `long-term` and `short-term memory`. It leverages `multimodal guidance` (text and image) to retrieve historical context, enabling the learning of `long-range dependencies` without reliance on 3D priors, handcrafted heuristics, or modifications to the core `DiT` framework. This design achieves `minute-level temporal consistency` with `linear computational complexity`.
2.  **Direct Forcing:** An efficient single-step approximating strategy that aligns training with inference. Built on `rectified flow`, it curtails `error propagation` and `exposure bias` in `autoregressive video generation` without incurring additional training cost, computational overhead, or requiring `distribution distillation`.

    Experimental results, including quantitative metrics and human evaluations, demonstrate that the combined approach significantly reduces `error accumulation`, enhances `identity-preservation consistency` (both subject and background), and alleviates `exposure bias`. This leads to `state-of-the-art performance` in `long-term consistency` metrics and overall quality. These contributions pave the way for more practical and reliable `next-generation long-form video generation models`.

## 7.2. Limitations & Future Work

The authors acknowledge the following limitations and suggest future research directions:

*   **MemoryPack (SemanticPack):** While achieving linear computational complexity and efficient training/inference, `SemanticPack` (which is part of `MemoryPack`) still `introduces artifacts in highly dynamic scenarios`. Its ability to maintain `long-term consistency in hour-long videos` (beyond minute-level) remains limited. Future work could explore more robust memory mechanisms for extreme dynamism and even longer durations.
*   **Direct Forcing:** It adopts a simple and efficient `single-step approximation strategy`. However, its effectiveness is `highly dependent on the pre-trained model`. As a result, the current training process requires `multiple stages` (teacher forcing followed by fine-tuning with Direct Forcing). Whether this fitting strategy can be integrated into a `single-stage pipeline` remains an open question and a direction for future research.

## 7.3. Personal Insights & Critique

This paper offers a compelling and well-structured approach to tackling the persistent challenges in long-form video generation. My personal insights and critiques are as follows:

*   **Innovation and Practicality:** The dual approach of `MemoryPack` and `Direct Forcing` is quite innovative. `MemoryPack`'s ability to dynamically integrate multimodal guidance for `long-term coherence` with `linear complexity` is a significant step forward, moving beyond static, local context windows. `Direct Forcing` intelligently leverages `rectified flow` to address the `training-inference mismatch` without the typical overheads of `Student Forcing` or distillation, which is a practical and elegant solution. The `RoPE Consistency` mechanism, while briefly described, is a clever addition to maintain cross-segment positional awareness.

*   **Potential for Transferability:** The `MemoryPack` concept of learnable, multimodal, and dynamic context retrieval could potentially be transferred to other sequence generation tasks that suffer from similar `long-range dependency` issues, such as `long-form text generation`, `audio generation`, or even `robot skill learning` where long-term task coherence is crucial. `Direct Forcing`'s principle of aligning training and inference through efficient approximation could also be beneficial for other `autoregressive generative models` beyond video.

*   **Critique and Areas for Improvement:**
    *   **Single-Stage Training for Direct Forcing:** The current necessity for `multi-stage training` for `Direct Forcing` (teacher forcing followed by fine-tuning) is a practical limitation. While the authors acknowledge this, a truly seamless, `single-stage training` pipeline would be more desirable and potentially simplify the adoption of the method. The dependence on a well-pretrained backbone also means `Direct Forcing` might not be a plug-and-play solution for all models.
    *   **Complexity of MemoryPack:** While `SemanticPack` achieves `linear complexity`, the full `MemoryPack` also includes `FramePack`. The paper describes `FramePack` as using a "fixed compression scheme." It would be beneficial to explicitly state the total computational complexity of `MemoryPack` in combination with the `MM-DiT` backbone more clearly. The `linear complexity` claim for `SemanticPack` is strong, but how it interacts with `FramePack` and the overall `DiT` architecture's efficiency for the whole system could be elaborated.
    *   **Depth of RoPE Consistency Explanation:** The explanation for `RoPE Consistency` could be more detailed for a beginner. Specifically, how assigning the image the "initial index of the entire video" practically translates into preserving cross-segment positional information via `RoPE`'s relative encoding property. A small architectural diagram showing the input sequence with the image token's position might help.
    *   **Artifacts in Dynamic Scenarios:** The limitation regarding `artifacts in highly dynamic scenarios` for `MemoryPack` is important. This suggests that while `MemoryPack` improves `consistency`, it might still struggle with rapid, unpredictable changes in motion or scene content. Future work could investigate adaptive context weighting or more sophisticated motion prediction within the memory mechanism.

        Overall, this paper presents a robust and promising step towards generating high-quality, long-form videos, demonstrating both innovative methodological designs and strong empirical results. The identified limitations also provide clear and exciting avenues for future research in this rapidly evolving field.