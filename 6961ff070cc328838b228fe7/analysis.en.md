# 1. Bibliographic Information

## 1.1. Title
WorldMem: Long-term Consistent World Simulation with Memory

## 1.2. Authors
*   Zeqi Xiao (S-Lab, Nanyang Technological University)
*   Yushi Lan (S-Lab, Nanyang Technological University)
*   Yifan Zhou (S-Lab, Nanyang Technological University)
*   Wenqi Ouyang (S-Lab, Nanyang Technological University)
*   Shuai Yang (Wangxuan Institute of Computer Technology, Peking University)
*   Yanhong Zeng (Shanghai AI Laboratory)
*   Xingang Pan (S-Lab, Nanyang Technological University)

    The authors are affiliated with prominent academic institutions and research labs in Asia, including Nanyang Technological University (NTU) in Singapore, Peking University, and the Shanghai AI Laboratory. This indicates a strong research background in computer vision and artificial intelligence.

## 1.3. Journal/Conference
The paper specifies a publication date of April 16, 2025, and cites other works with 2025 publication dates. This, combined with its availability on arXiv, indicates it is a preprint manuscript submitted for peer review, likely to a top-tier computer vision or machine learning conference such as CVPR, ICCV, or NeurIPS. The reputation of these venues is very high, representing the cutting edge of AI research.

## 1.4. Publication Year
2025 (as listed on the preprint). The first version was submitted to arXiv in April 2025.

## 1.5. Abstract
The abstract introduces the problem of maintaining long-term consistency in world simulation, which is often hampered by the limited temporal context window of generative models. This limitation is particularly detrimental to 3D spatial consistency. To address this, the paper presents `WorldMem`, a framework that integrates a memory bank into the scene generation process. This memory bank stores "memory units," which consist of past visual frames and their associated states (e.g., camera poses and timestamps). `WorldMem` employs a novel "memory attention" mechanism that uses these states to effectively retrieve relevant information from the memory bank. This allows the model to accurately reconstruct scenes even after long periods or significant viewpoint changes. By including timestamps, the framework can model not only static scenes but also their dynamic evolution, enabling both perception and interaction within the simulated world. The effectiveness of `WorldMem` is validated through extensive experiments in both virtual (Minecraft) and real-world scenarios.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2504.12369
*   **PDF Link:** https://arxiv.org/pdf/2504.12369v3.pdf
*   **Publication Status:** This is a preprint available on arXiv. It has not yet been officially published in a peer-reviewed journal or conference proceedings as of the time of this analysis.

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem this paper addresses is the **lack of long-term consistency** in generative world simulators. Modern video generation models, while powerful, operate within a fixed "context window"—they can only consider a limited number of recent frames when generating the next one. Once a scene detail moves out of this window, the model effectively forgets it.

This leads to significant logical and visual inconsistencies. For example, as illustrated in the paper, an agent might navigate away from a location and, upon returning, find that the scene has completely changed, even though it should have remained static. The following figure from the paper highlights this issue.

![Figure 1: WoRLDMEM enables long-term consistent world generation with an integrated memory mechanism. (a) Previous world generation methods typically face the problem of inconsistent world due to limited temporal context window size. (b) WoRLDMEM empowers the agent to explore diverse and consistent worlds with an expansive action space, e.., crafting environments by placing objects like pumpkin light or freely roaming around. Most importantly, after exploring for a while and glancing back, we find the objects we placed are still there, with the inspiring sight of the light melting the surrounding snow, testifying to the passage of time. Red and green boxes indicate scenes that should be consistent.](images/1.jpg)
*该图像是示意图，展示了在没有记忆机制的情况下（左侧）和采用记忆机制的情况下（右侧）进行世界生成的对比。左侧场景显示对象放置后并未在重新查看时保持一致，而右侧则通过记忆机制确保了环境的一致性，体现了时间的流逝和动态变化。*

This "amnesia" is a critical bottleneck for many applications:
*   **Autonomous Navigation:** A self-driving car or robot needs a consistent internal model of its environment to navigate reliably.
*   **Interactive Entertainment:** For a virtual world or game to be immersive, it must obey consistent rules. Objects should not randomly appear or disappear.
*   **Scientific Simulation:** Simulating physical or biological processes requires strict adherence to continuity over time.

    Previous attempts to solve this have fallen into two categories, each with its own drawbacks:
1.  **Geometric-based methods:** These explicitly build a 3D representation (like a mesh or a Neural Radiance Field) of the world. While this enforces consistency, it is inflexible. Modifying a pre-built 3D world (e.g., to simulate dynamic events) is difficult, and these methods can struggle with large, unbounded scenes.
2.  **Geometric-free methods:** These use implicit representations, such as abstract feature vectors, to store memory. However, these abstract memories often lose the fine-grained visual detail needed to reconstruct a scene accurately.

    The paper's innovative entry point is to create a memory system that is **both detailed and flexible**. Instead of abstract features, `WorldMem` stores the actual (latent) visual data of past frames and, crucially, augments this data with explicit **state information**: camera pose (location and orientation) and a timestamp. This allows the model to reason about *where* and *when* a memory occurred, enabling it to retrieve and use the right information to ensure consistency.

## 2.2. Main Contributions / Findings
The paper's primary contributions are:

1.  **The `WorldMem` Framework:** A novel architecture for long-term consistent world simulation that integrates an external memory bank with a state-of-the-art video diffusion model.

2.  **State-Aware Memory Mechanism:** This is the core technical innovation. It consists of:
    *   A **memory bank** that stores tokenized past frames along with their precise pose and timestamp.
    *   A **state-aware memory attention** module that allows the model to query the memory bank using spatial and temporal cues, retrieving relevant visual information to maintain consistency.
    *   A **relative state embedding** design that simplifies the model's task of reasoning about viewpoint changes.

3.  **Modeling of Dynamic Worlds:** By incorporating timestamps into the memory states, `WorldMem` can not only reconstruct static scenes but also model their evolution over time. For example, it can learn that snow melts or plants grow, and reflect these changes consistently when a location is revisited.

4.  **Demonstrated Effectiveness:** The paper provides strong empirical evidence through experiments on a custom `Minecraft` benchmark and the real-world `RealEstate10K` dataset. The results show that `WorldMem` significantly outperforms previous methods in maintaining 3D spatial consistency and generating high-fidelity, long-duration videos.

    In essence, `WorldMem` presents a scalable and effective solution to the "amnesia" problem in generative models, paving the way for more realistic, interactive, and consistent simulated worlds.

---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that learn to create data by reversing a noise-adding process. The core idea involves two steps:

1.  **Forward Process (Fixed):** This process gradually adds a small amount of Gaussian noise to a data sample (e.g., an image) over a series of $T$ steps. By the end, the original image is transformed into pure, unstructured noise. This process is mathematically defined and does not involve any learning.

2.  **Reverse Process (Learned):** The model learns to reverse this process. At each step $t$, it takes a noisy image and predicts the noise that was added to it (or, equivalently, predicts the slightly less noisy image from the previous step `t-1`). By starting with random noise and iteratively applying this learned denoising function for $T$ steps, the model can generate a completely new, clean data sample.

    `WorldMem` is built on a video diffusion model, which extends this principle to generate sequences of images (video frames).

### 3.1.2. Transformers & The Attention Mechanism
The **Transformer** is a neural network architecture that has become dominant in natural language processing and is increasingly used in computer vision. Its key innovation is the **attention mechanism**, which allows the model to weigh the importance of different parts of the input data when processing a specific part.

The most common form is **scaled dot-product attention**. For a given "query" element, it computes a score against a set of "key" elements. These scores are then scaled, converted into probabilities (weights) via a softmax function, and used to create a weighted sum of corresponding "value" elements.

The formula for attention is:
\$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\$
*   $Q$ (Query): A matrix representing the set of elements we are currently processing.
*   $K$ (Key): A matrix representing the set of elements we are paying attention to.
*   $V$ (Value): A matrix representing the content of the elements we are paying attention to.
*   $d_k$: The dimension of the keys, used for scaling to prevent the dot products from becoming too large.
*   `softmax`: A function that converts the scores into a probability distribution, ensuring the weights sum to 1.

    In `WorldMem`, this mechanism is crucial for allowing the current frame being generated (the query) to "look back" at the memory bank (the keys and values) and pull in relevant visual information.

### 3.1.3. Diffusion Transformers (DiT)
Traditional diffusion models often used a U-Net architecture as the denoising network. A **Diffusion Transformer (DiT)**, as proposed by Peebles and Xie (2023), replaces the U-Net with a Transformer. In this setup, the input (a noisy image) is first broken down into a sequence of patches. These patches are then treated as "tokens," similar to words in a sentence, and fed into a Transformer. The `DiT` architecture has proven to be highly scalable and effective, forming the backbone of many modern high-performance generative models, including the one used in `WorldMem`.

### 3.1.4. Autoregressive Generation
Autoregressive models generate data sequentially, where each new piece of data is conditioned on the previously generated pieces. In language, this means predicting the next word based on the words that came before it. In video, it means generating the next frame (or a small chunk of frames) based on the preceding frames. This is a natural fit for creating long videos, as it can theoretically extend a sequence indefinitely, unlike models that generate a whole fixed-length video at once.

## 3.2. Previous Works

### 3.2.1. Diffusion Forcing (DF)
`Diffusion Forcing` is a technique proposed by Chen et al. (2025) that enables stable, long-term autoregressive video generation with diffusion models. Traditional video diffusion models apply the same noise level to all frames in a context window during denoising. This "full-sequence" paradigm is not well-suited for autoregressive generation, where some frames are known (clean history) and others are unknown (noisy future).

`DF` introduces a **per-frame noise-level** denoising paradigm. During training and inference, each frame can have its own independent noise level. For autoregressive generation, this means the past (context) frames can be set to a very low noise level (almost clean), while the future frame to be generated is set to a high noise level. This provides a clean and effective way to extend videos frame by frame, making it a strong baseline for `WorldMem`.

### 3.2.2. Approaches to Consistent World Simulation
The paper categorizes prior work into two main types:

*   **Geometric-based Methods:** These methods create an explicit 3D or 4D representation of the world to enforce consistency. Examples include `Gen3C` (Ren et al., 2025) and `Viewcrafter` (Yu et al., 2024c), which might use techniques like multi-view synthesis or build explicit 3D assets.
    *   **Pro:** Guarantees strong geometric consistency.
    *   **Con:** Inflexible. It's hard to change the world once it's reconstructed (e.g., adding dynamic events) and can be computationally expensive for large scenes.

*   **Geometric-free Methods:** These methods rely on implicit learning without building an explicit 3D model.
    *   `StreamingT2V` (Henschel et al., 2024) and `SlowFastGen` (Hong et al., 2024) use abstract feature representations (like LoRA modules) to store memory. This is memory-efficient but often loses the fine-grained visual detail needed for high-fidelity reconstruction.
    *   Other methods like those by Alonso et al. (2025) achieve consistency by overfitting to specific, predefined environments (e.g., a single game map), which limits their scalability and generalizability.

## 3.3. Technological Evolution
The field of generative modeling has progressed rapidly:
1.  **Image Generation:** Models like GANs and early diffusion models focused on generating single, static images.
2.  **Short Video Generation:** Techniques were extended to generate short, coherent video clips, often using architectures that process all frames at once.
3.  **Long Video Generation:** The challenge of extending video length led to autoregressive approaches like `Diffusion Forcing`, but these still suffered from the limited context window, causing long-term inconsistencies.
4.  **Consistent World Simulation:** This is the current frontier, aiming not just to generate a long video, but to create an interactive and logically consistent *world*. This requires solving the "amnesia" problem.

    `WorldMem` fits into this latest stage. It directly tackles the consistency problem by introducing an explicit, long-term memory mechanism that is both detailed and flexible, representing a significant step beyond simple autoregressive generation.

## 3.4. Differentiation Analysis
`WorldMem`'s core innovation lies in *how* it formulates its memory, differentiating it from prior work:

*   **vs. Geometric-based Methods:** `WorldMem` is **geometry-free**, making it more flexible. It does not require explicit 3D reconstruction, so it can more easily model dynamic events and is not limited by the complexities of building and rendering large 3D scenes.

*   **vs. Abstract Memory Methods (`SlowFastGen`, etc.):** Instead of storing abstract, compressed features, `WorldMem` stores **token-level visual information** from past frames. This retains much more detail, enabling high-fidelity reconstruction.

*   **vs. Simple Autoregressive Models (`DF`):** While built on `DF`, `WorldMem` adds the crucial **external memory bank**. `DF` alone will still forget scenes that scroll out of its fixed context window. `WorldMem` can recall scenes from hundreds of frames in the past.

*   **The Key Differentiator:** The most unique aspect is the **state-aware attention**. `WorldMem` doesn't just store past images; it stores them with their **pose and timestamp**. This state information is embedded and used directly within the attention mechanism, allowing the model to perform explicit spatiotemporal reasoning. It can ask, "What did the world look like from *this specific location* at *that specific time*?" This is far more powerful than relying on ambiguous visual similarity alone.

    ---

# 4. Methodology

## 4.1. Principles
The central principle of `WorldMem` is to overcome the finite context window of autoregressive video models by equipping them with a long-term, external memory. The model should be able to consult this memory to ensure that when it revisits a location, the generated scene is consistent with what was seen before, even if the gap in time or viewpoint is large.

The intuition is that for generating the current view, only a small subset of all past experiences is relevant. The key is to effectively identify and retrieve this relevant subset and integrate it into the generation process. `WorldMem` achieves this through a **state-aware memory mechanism**, where memory retrieval and conditioning are guided not just by visual content but also by explicit spatial (pose) and temporal (timestamp) information.

The overall architecture is depicted in the figure below.

![Figure 2: Comprehensive overview of WoRLDMEM. The framework comprises a conditional diffusion transformer integrated with memory blocks, with a dedicated memory bank storing memory units from previously generated content. By retrieving these memory units from the memory bank and incorporating the information by memory blocks to guide generation, our approach ensures long-term consistency in world simulation.](images/2.jpg)
*该图像是示意图，展示了WorldMem框架的架构概览及其各部分功能。图中包含条件扩散变换器（DiT Block）、记忆块和记忆库，说明了如何从记忆库中获取样本以指导生成过程。图中还展示了状态嵌入的生成方式，包括姿态和时间戳的处理，以及不同输入噪声级别的比较。这些设计确保了长时间一致的世界模拟。*

## 4.2. Core Methodology In-depth (Layer by Layer)

The `WorldMem` framework is built upon a baseline interactive world simulator, which it enhances with a memory bank and a state-aware memory attention mechanism.

### 4.2.1. Baseline: Interactive World Simulator
The foundation of `WorldMem` is an autoregressive video generator that can be controlled by external actions.

*   **Architecture:** The model uses a **Conditional Diffusion Transformer (`CDiT`)**. The input video frames are first encoded into a latent space by a Variational Autoencoder (VAE) and then divided into patches, which are treated as a sequence of tokens. These tokens are processed by a series of `DiT` blocks. Each block contains spatial attention (for processing information within a single frame) and temporal attention (for processing information across frames). The temporal attention is **causal**, meaning a frame can only attend to itself and preceding frames.

*   **Autoregressive Generation:** To generate long videos, the model employs the **Diffusion Forcing (`DF`)** paradigm. This allows for stable frame-by-frame prediction by assigning different noise levels to different frames. During generation, the past context frames are kept nearly clean, while the new frame to be generated starts from pure noise and is iteratively denoised.

*   **Interaction:** The simulator is made interactive by conditioning on **action signals**. An action (e.g., "move forward," "turn left") is represented as a vector. This vector is projected into an embedding space using a Multi-Layer Perceptron (MLP) and then injected into the temporal blocks of the `DiT` using **Adaptive Layer Normalization (`AdaLN`)**. This allows the user's commands to influence the generation of the next frame.

    While this baseline can generate long, interactive videos, it will eventually "forget" scenes that scroll past its limited temporal context window, leading to inconsistency.

### 4.2.2. Memory Representation and Retrieval
To solve the forgetting problem, `WorldMem` introduces an external memory.

*   **Memory Representation:** The memory bank is a set of **memory units**. Each unit is a tuple $(\mathbf{x}_i^m, \mathbf{p}_i, t_i)$:
    *   $\mathbf{x}_i^m$: The latent representation (tokens) of a previously generated frame, compressed by the VAE. This retains rich visual detail.
    *   $\mathbf{p}_i$: The camera pose associated with the frame, a 5D vector containing position (x, y, z) and orientation (pitch, yaw).
    *   $t_i$: The timestamp of when the frame was generated.

*   **Memory Retrieval (Algorithm 1):** At each generation step, the model needs to select a small, relevant subset of memory units to condition on. Since the memory bank can be large, an efficient retrieval strategy is needed. `WorldMem` uses a greedy matching algorithm based on a "confidence score".

    The algorithm proceeds as follows:
1.  **Compute Confidence Score:** For the current state $( \mathbf { x } _ { c } , \mathbf { p } _ { c } , t _ { c } )$ and every memory unit $i$ in the bank, calculate a confidence score $\alpha_i$. This score combines spatial and temporal proximity.
    *   **Field-of-View (FOV) Overlap:** Calculate the overlap ratio $\mathbf{o}_i$ between the current camera's FOV and the memory camera's FOV. This is estimated using Monte Carlo sampling (randomly sampling points in space and checking if they fall into both views).
    *   **Time Difference:** Calculate the absolute time difference $\mathbf{d}_i = |t_i - t_c|$.
    *   The final confidence score is a weighted combination:
        \$
        \pmb { \alpha } = \mathbf { o } \cdot w _ { o } - \mathbf { d } \cdot w _ { t }
        \$
        where $w_o$ and $w_t$ are weights. This score prioritizes memories that have high spatial overlap and are temporally close.

2.  **Greedy Selection with Similarity Filtering:** To select $L_M$ memory frames:
    *   Initialize an empty selection set $S$.
    *   In a loop, select the memory unit $i^*$ with the highest confidence score $\alpha_{i^*}$.
    *   Add $i^*$ to the set $S$.
    *   To ensure diversity and avoid redundant information, remove all other memory units $j$ from consideration that are highly similar to $i^*$ (e.g., their FOV overlap is above a threshold `tr`).
    *   Repeat until $L_M$ units are selected.

### 4.2.3. State-aware Memory Conditioning
Once the relevant memory units are retrieved, their information must be integrated into the generation process. This is done through a special cross-attention mechanism within the `DiT` blocks, which the paper calls **memory blocks**.

*   **State Embedding:** First, the state information (pose and timestamp) for both the current (query) frames and the memory (key) frames must be converted into embeddings.
    *   The final state embedding $\mathbf{E}$ is the sum of a pose embedding and a time embedding:
        \$
        \mathbf { E } = G _ { p } ( \mathbf { P E } ( \mathbf { p } ) ) + G _ { t } ( \mathbf { S E } ( t ) )
        \$
        *   $\mathbf{p}$: The 5D pose vector.
        *   $\mathbf{PE}(\mathbf{p})$: A **Plücker embedding** of the pose. This technique represents each camera ray as a 6D vector, effectively converting the pose into a dense feature map that encodes spatial information for every pixel.
        *   $t$: The timestamp.
        *   $\mathbf{SE}(t)$: A standard sinusoidal embedding of the timestamp.
        *   $G_p$ and $G_t$: MLPs that project the pose and time features into a shared embedding dimension.

*   **State-aware Memory Attention:** This is a cross-attention layer where the queries come from the current input frames, and the keys/values come from the retrieved memory frames. The key innovation is to inject the state embeddings into the attention calculation.
    *   Let $\mathbf{X}_q$ be the latent tokens of the input frames and $\mathbf{X}_k$ be the latent tokens of the memory frames.
    *   Let $\mathbf{E}_q$ and $\mathbf{E}_k$ be their corresponding state embeddings.
    *   The state information is added to the visual tokens before they are projected into queries and keys:
        \$
        \tilde { \mathbf { X } } _ { q } = \mathbf { X } _ { q } + \mathbf { E } _ { q } , \quad \tilde { \mathbf { X } } _ { k } = \mathbf { X } _ { k } + \mathbf { E } _ { k }
        \$
    *   The cross-attention is then computed as:
        \$
        { \bf X } ^ { \prime } = \mathrm { C r o s s A t t n } ( Q = p _ { q } ( \tilde { \bf X } _ { q } ) , { \cal K } = p _ { k } ( \tilde { \bf X } _ { k } ) , { \cal V } = p _ { v } ( { \bf X } _ { k } ) )
        \$
        *   $p_q, p_k, p_v$ are learnable projection matrices.
        *   Notice that the state embedding is **not** added to the values ($\mathcal{V}$). This is because the values contain the raw visual content that needs to be copied, while the queries and keys are used for matching and alignment. Adding state information to Q and K helps the model find the correct correspondence based on spatiotemporal context.

*   **Relative State Formulation:** To make learning easier, the model uses relative states instead of absolute ones. For each query, its own pose is treated as the origin (identity matrix) and its timestamp as zero. The poses and timestamps of the memory frames are then transformed to be relative to this query frame. This means the model doesn't have to learn absolute positions in a global coordinate system, but rather the simpler task of how to transform information from one relative viewpoint to another.

### 4.2.4. Incorporating Memory into the Pipeline
*   **Noise Levels:** During training and inference, the retrieved memory frames are treated as clean context. They are assigned the lowest noise level, $k_{min}$, similar to the past frames in the standard autoregressive setup. The frames currently being generated are assigned the highest noise level, $k_{max}$.

*   **Temporal Attention Mask:** To ensure that the memory frames only provide information to the main context window and do not interact with each other, a special attention mask is used in the temporal attention layers. The mask is defined as:
    \$
    A _ { \mathrm { mask } } ( i , j ) = \left\{ { \begin{array} { l l } { 1 , } & { i \leq L _ { M } { \mathrm { and } } j = i } \\ { 1 , } & { i > L _ { M } { \mathrm { and } } j \leq i } \\ { 0 , } & { { \mathrm { otherwise } } } \end{array} } \right.
    \$
    *   $L_M$ is the number of memory frames, which are prepended to the input sequence.
    *   This mask enforces two rules:
        1.  A memory frame ($i \le L_M$) can only attend to itself. This prevents memory frames from corrupting each other.
        2.  A regular context frame ($i > L_M$) can attend to all preceding frames (both memory and other context frames), maintaining the causal structure.

            ---

# 5. Experimental Setup

## 5.1. Datasets
*   **MineDojo (Minecraft):** A large-scale framework for building embodied AI agents in Minecraft. The authors used it to create a custom dataset featuring diverse environments (plains, savannas, deserts, ice plains), a wide range of agent actions (movement, camera control), and environmental interactions (e.g., weather changes, placing objects). The training set consists of ~12,000 long videos, each 1500 frames long. This rich, interactive environment is ideal for testing long-term consistency and dynamic event modeling.

*   **RealEstate10K:** A large dataset of real-world videos scraped from YouTube, primarily consisting of indoor and outdoor real estate tours. Crucially, it comes with pre-computed camera pose annotations for each frame. The training set has ~65,000 short video clips. This dataset is used to validate the model's performance on real-world scenes, focusing on view synthesis and consistency during camera motion.

## 5.2. Evaluation Metrics

To quantitatively measure the quality and consistency of the generated videos, the paper uses three standard metrics.

### 5.2.1. PSNR (Peak Signal-to-Noise Ratio)
*   **Conceptual Definition:** PSNR measures the pixel-level fidelity of a generated image compared to a ground truth image. It quantifies the ratio between the maximum possible power of a signal (the maximum pixel value) and the power of corrupting noise (the error between the two images). A higher PSNR indicates that the generated image is closer to the ground truth at a pixel level.
*   **Mathematical Formula:**
    \$
    \text{PSNR} = 20 \cdot \log_{10}(\text{MAX}_I) - 10 \cdot \log_{10}(\text{MSE})
    \$
*   **Symbol Explanation:**
    *   $\text{MAX}_I$: The maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image).
    *   $\text{MSE}$: The Mean Squared Error between the ground truth image $I$ and the generated image $K$, calculated as `\frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2` for images of size $m \times n$.

### 5.2.2. LPIPS (Learned Perceptual Image Patch Similarity)
*   **Conceptual Definition:** LPIPS measures the perceptual similarity between two images. Unlike PSNR, which is sensitive to small pixel shifts, LPIPS is designed to align better with human judgment. It works by feeding both the generated and ground truth images through a pre-trained deep neural network (like VGG or AlexNet) and comparing their intermediate feature activations. If the features are similar, the images are considered perceptually similar. A lower LPIPS score indicates higher similarity.
*   **Mathematical Formula:**
    \$
    d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} || w_l \odot ( \hat{y}_{hw}^l - \hat{y}_{0hw}^l ) ||_2^2
    \$
*   **Symbol Explanation:**
    *   $x, x_0$: The two images being compared.
    *   $l$: Index of the layer in the deep network. The total distance is the sum over multiple layers.
    *   $\hat{y}^l, \hat{y}_0^l$: The feature activations from layer $l$ for images $x$ and $x_0$, respectively.
    *   $H_l, W_l$: The spatial dimensions of the feature maps at layer $l$.
    *   $w_l$: A learned weight vector used to scale the contribution of each channel.

### 5.2.3. rFID (reconstruction Fréchet Inception Distance)
*   **Conceptual Definition:** FID is a standard metric for evaluating the quality of generative models. It measures the distance between the distribution of features from real images and the distribution of features from generated images. These features are extracted from a pre-trained InceptionV3 network. A lower FID suggests that the distribution of generated images is closer to that of real images, indicating higher realism and diversity. **rFID** is a variant used for reconstruction tasks, where it computes the FID score between a set of generated frames and their corresponding ground truth frames.
*   **Mathematical Formula:**
    \$
    \text{FID}(x, g) = ||\mu_x - \mu_g||_2^2 + \text{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})
    \$
*   **Symbol Explanation:**
    *   $\mu_x, \mu_g$: The mean of the feature vectors for the real (or ground truth) and generated images, respectively.
    *   $\Sigma_x, \Sigma_g$: The covariance matrices of the feature vectors for the real and generated images.
    *   $\text{Tr}$: The trace of a matrix (the sum of the elements on the main diagonal).

## 5.3. Baselines
The paper compares `WorldMem` against several representative baselines.

*   **For the Minecraft benchmark:**
    *   **Full Seq. (Full Sequence):** A standard conditional `DiT` model trained with the full-sequence paradigm, where all frames in the context window share the same noise level. This approach is not suitable for long autoregressive rollout.
    *   **DF (Diffusion Forcing):** The same `CDiT` architecture but trained and inferred using the `DF` paradigm, enabling autoregressive generation. This serves as the direct baseline to show the improvement gained by adding the memory mechanism.

*   **For the RealEstate10K benchmark:**
    *   **CameraCtrl, TrajAttn, DFoT:** These are recent video generation models capable of camera control but lack explicit long-term memory mechanisms.
    *   **Viewcrafter:** A strong baseline that uses an **explicit 3D reconstruction** approach to maintain consistency. Comparing against `Viewcrafter` directly tests the performance of `WorldMem`'s geometry-free approach against a geometry-based one.

        ---

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Comparisons on Minecraft Benchmark
The Minecraft experiments are designed to test consistency in both short-term (within context) and long-term (beyond context) scenarios.

*   **Within Context Window:** This experiment tests self-contained consistency, where the agent moves away and returns to a starting point within the model's 16-frame context window.

    The following are the results from Table 1 of the original paper:

    <table>
    <thead>
    <tr>
    <th colspan="4">Within context window</th>
    </tr>
    <tr>
    <th>Methods</th>
    <th>PSNR ↑</th>
    <th>LPIPS ↓</th>
    <th>rFID ↓</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Full Seq.</td>
    <td>20.14</td>
    <td>0.0691</td>
    <td>13.87</td>
    </tr>
    <tr>
    <td>DF</td>
    <td>24.11</td>
    <td>0.0094</td>
    <td>13.88</td>
    </tr>
    <tr>
    <td>Ours</td>
    <td>25.98</td>
    <td>0.0072</td>
    <td>13.73</td>
    </tr>
    </tbody>
    </table>

**Analysis:** `WorldMem` ("Ours") achieves the best scores on all metrics. The `Full Seq.` baseline performs poorly, showing it struggles with consistency even over short durations. `DF` improves significantly over `Full Seq.`, but `WorldMem` still provides a noticeable boost, indicating that the memory mechanism helps improve reconstruction quality even for recently seen frames. The visual results in Figure 4 confirm this, showing `WorldMem` produces a much sharper and more accurate reconstruction upon returning to the start.

![Figure 4: Within context window evaluation. The motion sequence involves turning right and returning to the original position, showing selfcontained consistency.](images/4.jpg)
*该图像是图表，展示了不同场景下的自我一致性评估。第一行显示了完整序列的视图，第二行为DF方法的视图，第三行为我们的方法。通过“初始化”、“向右转”和“向左转”的场景变化，展示了在不同观察角度下的场景重建效果。*

*   **Beyond Context Window:** This is the most critical test. The model generates 100 frames after an initial history of 600 frames, and consistency is checked against a location seen in that initial history, far outside the 8-frame context window.

    The following are the results from Table 1 of the original paper:

    <table>
    <thead>
    <tr>
    <th colspan="4">Beyond context window</th>
    </tr>
    <tr>
    <th>Methods</th>
    <th>PSNR ↑</th>
    <th>LPIPS ↓</th>
    <th>rFID ↓</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Full Seq.</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    </tr>
    <tr>
    <td>DF</td>
    <td>17.32</td>
    <td>0.4376</td>
    <td>51.28</td>
    </tr>
    <tr>
    <td>Ours</td>
    <td>23.98</td>
    <td>0.1429</td>
    <td>15.37</td>
    </tr>
    </tbody>
    </table>

**Analysis:** The results are stark. The `DF` baseline completely fails, with very poor PSNR/LPIPS scores and a high `rFID`, indicating it generates an inconsistent scene that is perceptually different from the ground truth. In contrast, `WorldMem` maintains high scores, demonstrating its ability to accurately recall and reconstruct scenes from its long-term memory. The `rFID` score is particularly telling: `WorldMem`'s `rFID` of 15.37 is nearly as good as its within-context performance, while `DF`'s `rFID` explodes to 51.28. Figure 5 visually confirms this drastic difference.

![Figure 5: Beyond context window evaluation. Diffusion-Forcing suffers inconsistency over time, while ours maintains quality and recovers past scenes.](images/5.jpg)
*该图像是比较不同时间帧生成效果的示意图。DF方法在时间上存在不一致性，而我们的方法能保持质量，并准确恢复过去的场景。显示了Frame 0、Frame 50和Frame 100的对比结果。*

*   **Qualitative & Dynamic Results:** Figure 3 showcases the model's ability to handle diverse environments and dynamic events. The bottom row is particularly impressive, showing that `WorldMem` can remember that wheat was planted and correctly render it as having grown over time when the location is revisited. This demonstrates that the timestamp conditioning allows the model to capture the temporal evolution of the world, not just static consistency.

    ![Figure 3: Qualitative results. We showcase WoRLDMEM's capabilities through two sets of examples. Top: A comparison with Ground Truth (GT). WoRLDMEM accurately models diverse dynamics (e.g., rain) by conditioning on 600 past frames, ensuring temporal consistency. Bottom: Interaction with the world. Objects like hay in the desert or wheat in the plains persist over time, with wheat visibly growing. For the best experience, see the supplementary videos.](images/3.jpg)
    *该图像是图表，展示了WoRLDMEM框架在不同场景下的表现，包括与真实场景（GT）的对比。上半部分展示了在多种动态条件下的表现，例如雨天，确保了时间上的一致性；下半部分则展示了与环境的互动过程，例如在沙漠中放置干草等。*

### 6.1.2. Comparisons on Real Scenarios (RealEstate10K)
This experiment tests the model on real-world videos. The task is to complete a "loop closure" trajectory, where the camera returns to its starting pose after a long path.

The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>Methods</th>
<th>PSNR ↑</th>
<th>LPIPS ↓</th>
<th>rFID ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>CameraCtrl (He et al., 2024)</td>
<td>13.19</td>
<td>0.3328</td>
<td>133.81</td>
</tr>
<tr>
<td>TrajAttn (Xiao et al., 2024)</td>
<td>14.22</td>
<td>0.3698</td>
<td>128.36</td>
</tr>
<tr>
<td>Viewcrafter (Yu et al., 2024c)</td>
<td>21.72</td>
<td>0.1729</td>
<td>58.43</td>
</tr>
<tr>
<td>DFoT (Song et al., 2025)</td>
<td>16.42</td>
<td>0.2933</td>
<td>110.34</td>
</tr>
<tr>
<td>Ours</td>
<td>23.34</td>
<td>0.1672</td>
<td>43.14</td>
</tr>
</tbody>
</table>

**Analysis:** `WorldMem` outperforms all baselines, including `Viewcrafter`, which uses explicit 3D reconstruction. This is a significant result, suggesting that `WorldMem`'s flexible, geometry-free memory approach can achieve superior consistency and fidelity compared to more rigid geometric methods, likely because it avoids errors that can accumulate during the 3D reconstruction and rendering process. Figure 6 shows `WorldMem` producing a final frame that is visually much more consistent with the first frame compared to the baselines.

![Figure 6: Results on RealEstate (Zhou et al., 2018). We visualize loop closure consistency over a full camera rotation. The visual similarity between the first and last frames serves as a qualitative indicator of 3D spatial consistency.](images/6.jpg)
*该图像是一个示意图，展示了不同方法在房地产场景中的表现，包括CameraCut、Viewcrafter、DFoT和我们的方案。每列显示在相同视角下的结果，以比较各方法在3D空间一致性表现上的差异。*

## 6.2. Ablation Studies / Parameter Analysis
The paper conducts several ablation studies to validate the importance of each component of `WorldMem`.

*   **Embedding Designs (Table 2):** This study compares different ways of encoding the pose information.
    *   **Dense vs. Sparse:** Using a dense Plücker embedding is far superior to a sparse pose representation, highlighting the need for rich spatial cues.
    *   **Relative vs. Absolute:** Using relative pose embeddings provides a significant boost, especially in `rFID` (from 29.34 to 15.37). This confirms that teaching the model to reason about relative viewpoint changes is much more effective than forcing it to learn a global coordinate system. Figure 7 shows that the model with absolute embeddings degrades significantly after 100 frames, while the full model with relative embeddings remains stable.

*   **Memory Retrieval Strategy (Table 3):**
    *   **Random retrieval** performs terribly, confirming that targeted retrieval is essential.
    *   Adding **confidence-based filtering** (using FOV overlap and time) brings a massive improvement.
    *   Further adding **similarity filtering** to ensure diverse memory samples provides an additional boost. This validates the paper's full retrieval algorithm.

*   **Time Condition (Table 6):** This ablation removes the timestamp from the memory state and retrieval process. The results show a drop in performance, especially on a curated test set with dynamic events. This confirms that the time condition is crucial for the model to correctly reason about a world that changes over time, as shown qualitatively in Figure 8.

    ![Figure 8: Results w/o and w/ time condition. Without timestamps, the model fails to differentiate memory units from the same location at different times, causing errors. With time conditioning, it aligns with the updated world state, ensuring consistency.](images/8.jpg)
    *该图像是一个示意图，展示了在没有时间条件下的结果。上方为初始化、放置稻草、环绕走动和回望的场景，下方为有时间条件下的对比。未使用时间戳时，模型无法区分同一地点在不同时间的记忆单元，导致错误。包涵时间条件后，模型能够与更新的世界状态对齐，确保一致性。*

*   **Memory Context Length (Table 7):** This study varies the number of memory frames ($L_M$) used for conditioning. Performance improves up to a length of 8, but then degrades at 16. This suggests an optimal trade-off: too few memory frames provide insufficient context, while too many may introduce noise or conflicting information, hurting retrieval precision.

*   **Pose Prediction (Table 8):** In a real interactive scenario, the ground truth pose for the next frame is unknown. This experiment shows that using a lightweight module to predict the next pose from the previous frame and action yields performance that is only slightly worse than using the ground truth pose. This demonstrates the practical viability of the system in a real-world setting where only action commands are available.

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully introduces `WorldMem`, a novel framework that addresses the critical challenge of long-term consistency in generative world simulations. By augmenting a state-of-the-art video diffusion model with an external memory bank, `WorldMem` overcomes the limitations of fixed-size context windows. The core innovation is its **state-aware memory attention mechanism**, which leverages explicit pose and timestamp information to accurately retrieve and condition on relevant past experiences. This enables the model to reconstruct previously seen scenes with high fidelity, even across significant temporal and viewpoint gaps. Furthermore, the inclusion of timestamps allows `WorldMem` to model dynamic worlds that evolve over time. Extensive experiments on both virtual and real-world benchmarks demonstrate that `WorldMem` significantly outperforms existing methods, establishing a new state-of-the-art in consistent and interactive world generation.

## 7.2. Limitations & Future Work
The authors candidly acknowledge several limitations and areas for future research:

*   **Imperfect Memory Retrieval:** The current retrieval strategy relies on heuristics like FOV overlap. This can fail in corner cases, such as when a previously seen location is temporarily occluded by an obstacle. A more robust, perhaps learned, retrieval mechanism could be beneficial.
*   **Limited Interaction Realism:** While the model supports actions, the diversity and realism of interactions are still limited. Future work aims to extend the framework to more complex and realistic interactions in real-world scenarios.
*   **Memory Scalability:** The memory bank currently grows linearly with the length of the simulation. For extremely long sequences, this could become a bottleneck in terms of storage and retrieval efficiency. Developing hierarchical memory or summarization techniques to manage memory growth is a key future direction.

## 7.3. Personal Insights & Critique
This paper presents a very elegant and powerful solution to a fundamental problem in generative AI.

**Positive Insights:**
*   **The Power of Explicit State:** The key takeaway for me is the effectiveness of integrating explicit state information (pose, time) directly into the attention mechanism. While many models rely on the network to implicitly learn spatial relationships from visual data alone, `WorldMem` shows that providing these cues explicitly makes the learning task much easier and the results far more robust. The use of relative embeddings is a particularly clever design choice that simplifies this reasoning process.
*   **Flexible yet Detailed Memory:** The framework strikes an excellent balance. It avoids the rigidity of geometric methods while retaining far more detail than abstract feature-based memories. Storing latent tokens is a sweet spot that is both information-rich and computationally manageable.
*   **Scalability and Practicality:** The design is built on standard components like Transformers and cross-attention, making it highly scalable with modern hardware. The ablation study on pose prediction also shows a clear path toward practical deployment in interactive applications where ground truth states are unavailable.

**Critique and Potential Areas for Improvement:**
*   **Retrieval Mechanism:** As the authors note, the heuristic-based retrieval is a potential weak point. A future direction could involve a learned retriever, perhaps a small, separate network that learns to predict which memories will be most useful for a given query, moving beyond simple geometric overlap.
*   **Generalization of Dynamics:** The model's ability to show wheat growing or snow melting is impressive. However, this is likely interpolated from dynamics observed in the training data. The model's ability to generalize to novel, unseen dynamic processes is an open and challenging question. It might correctly simulate a fire spreading if it has seen fires before, but it would likely fail to simulate a completely new physical phenomenon.
*   **Compositionality:** The current memory is a flat list. A more advanced system might benefit from a more structured, compositional memory. For instance, being able to reason about individual objects and their states separately from the background scene could enable more complex and robust world modeling.

    Overall, `WorldMem` is a significant contribution that provides a strong foundation for the next generation of stateful, consistent, and interactive generative models. Its core ideas are likely to be highly influential and applicable not only to world simulation but also to related fields like robotics, embodied AI, and long-form content generation.