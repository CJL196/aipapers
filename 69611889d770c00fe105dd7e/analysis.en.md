# 1. Bibliographic Information

## 1.1. Title
Memory Forcing: Spatio-Temporal Memory for Consistent Scene Generation on Minecraft

The title clearly states the paper's core contribution: a method named **`Memory Forcing`** designed to improve scene generation in the Minecraft environment. It highlights the use of **`spatio-temporal memory`** (memory that considers both space and time) to achieve **`consistent`** scenes, which is a major challenge in long-duration generative tasks.

## 1.2. Authors
The authors are Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Their affiliations include The Chinese University of Hong Kong (Shenzhen), The University of Hong Kong, Voyager Research (Didi Chuxing), and Microsoft Research. The presence of researchers from both top-tier academic institutions and leading industry research labs suggests a strong combination of theoretical rigor and practical application focus.

## 1.3. Journal/Conference
The paper does not explicitly state its publication venue. However, the publication date is listed as October 2025. This suggests it is a preprint submitted to a future top-tier conference. Major AI and computer vision conferences like NeurIPS, ICML, ICLR, or CVPR are likely targets. Publishing in such venues indicates a high standard of peer review and significant impact in the field.

## 1.4. Publication Year
2025 (as per the preprint metadata).

## 1.5. Abstract
The abstract introduces the central problem in autoregressive video diffusion models for interactive scene generation, particularly in Minecraft: the trade-off between generating novel content and maintaining spatial consistency when revisiting locations. The authors argue that temporal-only memory fails to ensure long-term consistency, while adding spatial memory can harm the quality of new scene generation. To solve this, they propose **`Memory Forcing`**, a framework combining specialized training protocols with a **`geometry-indexed spatial memory`**. The key components are:
1.  **`Hybrid Training`**: Teaches the model to use temporal memory for exploration and spatial memory for revisits.
2.  **`Chained Forward Training`**: Uses model rollouts during training to encourage reliance on spatial memory for consistency.
3.  **`Geometry-indexed Spatial Memory`**: An efficient system using `Point-to-Frame Retrieval` and `Incremental 3D Reconstruction` to manage historical data.
    The abstract concludes that `Memory Forcing` achieves superior long-term consistency and generative quality while being computationally efficient.

## 1.6. Original Source Link
- **Original Source Link:** https://arxiv.org/abs/2510.03198
- **PDF Link:** https://arxiv.org/pdf/2510.03198v1.pdf
- **Publication Status:** This is a preprint available on arXiv. It has not yet undergone formal peer review for a conference or journal publication.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem this paper addresses is a fundamental dilemma in generative world models, especially those used for interactive environments like Minecraft. When an AI model generates a video sequence of gameplay autoregressively (frame by frame), it faces a trade-off:

1.  **Exploration vs. Consistency:** The model must be creative enough to generate new, unexplored areas of the world convincingly. At the same time, if the player revisits a location, the generated scene must be consistent with what was seen before. For example, if a player builds a structure, leaves, and comes back, the structure should still be there.

2.  **The Memory Bottleneck:** Autoregressive models can only look at a limited number of past frames (a `context window`) due to computational constraints. This creates a challenge:
    *   **Temporal-only Memory:** Using only the most recent frames as memory is good for smooth, short-term motion but fails to remember distant locations. This leads to inconsistency upon revisits (as shown in Figure 1b).
    *   **Spatial Memory:** Incorporating frames from past, spatially relevant locations can enforce consistency. However, if the model over-relies on this spatial memory when there isn't enough relevant history (e.g., in a brand-new area), its ability to generate high-quality new scenes degrades (as shown in Figure 1a).

        The paper's motivation is to create a framework that **resolves this trade-off**. The innovative idea is to "force" the model, through specialized training, to learn *when* to rely on recent temporal context (during exploration) and *when* to incorporate long-term spatial memory (during revisits).

        ![Figure 1: Two paradigms of autoregressive video models and their fail cases. (a) Long-term spatial memory models maintain consistency when revisiting areas yet deteriorate in new environments. (b) Temporal memory models excel in new scenes yet lack spatial consistency when revisiting areas.](images/1.jpg)
        *该图像是示意图，展示了依赖长期空间记忆和短期时间记忆的自回归视频扩散变换器在游戏场景生成中的应用。左侧图(a)显示如何通过长期空间记忆生成下一帧，而右侧图(b)则表明仅依赖短期时间记忆会导致失败。涉及的模型结构和记忆体之间的关系通过虚拟轨迹进行阐述。*

## 2.2. Main Contributions / Findings
The paper presents three main contributions encapsulated within the **`Memory Forcing`** framework:

1.  **A Novel Training Framework (`Memory Forcing`):** This framework is designed to teach a video diffusion model how to dynamically balance its use of temporal and spatial memory. It aims to achieve both high-quality new scene generation and long-term spatial consistency, solving the core trade-off problem.

2.  **Specialized Training Protocols:**
    *   **`Hybrid Training`:** This protocol uses two different types of gameplay data. On "exploration-heavy" data, it trains the model to use only temporal memory. On "revisit-heavy" data, it trains the model to use a hybrid of temporal and spatial memory. This teaches the model to adapt its memory usage based on the context.
    *   **`Chained Forward Training (CFT)`:** This technique extends standard training by using the model's own generated frames as input for subsequent predictions within a sequence. This simulates the error accumulation that happens during long-term inference and encourages the model to rely more on the stable, geometry-based spatial memory to correct for drift and maintain consistency.

3.  **An Efficient Spatial Memory System (`Geometry-indexed Spatial Memory`):** Instead of just storing past frames and searching through them based on camera pose (which is slow and inefficient), the paper proposes a more sophisticated system.
    *   It builds a coarse 3D model of the scene (`Incremental 3D Reconstruction`) as the game is played.
    *   To retrieve relevant past frames, it uses **`Point-to-Frame Retrieval`**: it identifies which 3D points are visible in the current view and retrieves the original frames from which those points were first seen. This is highly efficient and robust to changes in viewpoint.

        The key finding is that this combination of intelligent training and efficient memory management successfully improves both generative quality and long-term consistency in Minecraft, outperforming previous methods while being significantly faster and using less memory for retrieval.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Autoregressive Models
An autoregressive model is a type of generative model that produces a sequence of data points one at a time, where each new data point is conditioned on the ones that came before it. In the context of video generation, this means generating the next frame $x_t$ based on a history of previous frames $(x_{t-1}, x_{t-2}, ...)$. This sequential, step-by-step generation makes them a natural fit for modeling time-series data like video.

### 3.1.2. Diffusion Models
Diffusion models are a class of powerful generative models that have recently surpassed Generative Adversarial Networks (GANs) in image synthesis quality. They work in two phases:

1.  **Forward Process (Noise Addition):** A clean image is gradually corrupted by adding a small amount of Gaussian noise over many time steps. After a large number of steps, the image becomes pure noise. This process is a fixed mathematical procedure.
2.  **Reverse Process (Denoising):** The model learns to reverse this process. It takes a noisy image and a time step as input and predicts the noise that was added at that step. By repeatedly subtracting the predicted noise, the model can gradually transform a pure noise image into a clean, new image.

    The training objective for a diffusion model is typically to minimize the difference between the actual added noise $\epsilon$ and the model's predicted noise $\epsilon_\theta$:
\$
\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[ \|\epsilon - \epsilon_{\theta}(\alpha_t x_0 + \sigma_t \epsilon, t)\|^2 \right]
\$
where $x_0$ is the original image, $\epsilon$ is random noise, $t$ is the time step, and $\alpha_t, \sigma_t$ are schedule constants.

### 3.1.3. Diffusion Transformer (DiT)
A Diffusion Transformer, or `DiT`, is an architecture that replaces the commonly used U-Net backbone in diffusion models with a Transformer. In a `DiT`, an image is first broken down into a sequence of patches (tokens). The model then operates on these tokens, using the self-attention mechanism of Transformers to learn relationships between different parts of the image. This architecture has been shown to scale effectively with model size and compute, leading to state-of-the-art results in image generation. This paper uses a `DiT` backbone for its video generation model.

### 3.1.4. World Model
In reinforcement learning and AI, a "world model" is a model that learns the dynamics of an environment. It can predict future states of the environment given the current state and an action. For example, in a game, a world model could predict the next frame of the game given the current frame and the player's button press. Powerful world models, like the one this paper aims to build, can effectively simulate the environment, enabling planning, imagination, and better decision-making for an AI agent.

## 3.2. Previous Works

### 3.2.1. Diffusion Forcing
`Diffusion Forcing` (Chen et al., 2024) is a training technique for autoregressive video diffusion models. It allows for flexible conditioning by training the model on sequences containing a mix of clean (ground truth) and noisy frames. This helps bridge the gap between training (where models often see only clean past frames) and inference (where models must generate based on their own, potentially imperfect, past predictions). This paper builds on this framework for its autoregressive generation setup.

### 3.2.2. VPT and MineDojo
*   **Video PreTraining (VPT)** (Baker et al., 2022) is a large-scale dataset of unlabeled Minecraft gameplay videos. Models can be pre-trained on this dataset to learn the fundamental visual dynamics and action consequences of the Minecraft world. This paper uses `VPT` for its exploration-oriented training data.
*   **MineDojo** (Fan et al., 2022) is a comprehensive framework and dataset for building agents in Minecraft. It provides an environment for generating synthetic trajectories with specific properties. This paper uses `MineDojo` to create its "revisit-heavy" dataset for training long-term spatial consistency.

### 3.2.3. Baseline Models
*   **Oasis** (Decart et al., 2024) and **NFD** (Cheng et al., 2025) are recent autoregressive video models for Minecraft that demonstrate strong interactive capabilities. However, they primarily rely on short-term temporal memory within a fixed context window and thus lack long-term spatial consistency. They represent the "temporal-only" paradigm.
*   **WorldMem** (Xiao et al., 2025) is a model that explicitly incorporates long-term spatial memory. It stores past frames in a memory bank and retrieves them based on camera pose similarity (field-of-view overlap). While it improves consistency on revisits, the paper notes that it struggles with new scene generation and its retrieval mechanism becomes inefficient as the sequence length grows. It represents the "spatial-memory" paradigm.

### 3.2.4. 3D Reconstruction and Retrieval
*   **VGGT** (Wang et al., 2025a) is a Transformer-based model for visual geometry estimation. It can predict depth maps and camera poses from multiple views. This paper uses `VGGT` as the engine for its `Incremental 3D Reconstruction`.
*   **VMem** (Li et al., 2025a) is another related work that uses 3D geometry for memory in video generation. It uses a "surfel-indexed" view selection method. The current paper's `Geometry-indexed Spatial Memory` is inspired by this line of work, aiming for even greater efficiency and scalability.

## 3.3. Technological Evolution
The evolution of interactive world models, particularly for Minecraft, can be seen as a progression in memory management:

1.  **Stateless Models:** Early models generated frames based only on the immediate past frame and action, with no long-term memory.
2.  **Temporal Context Window Models:** Models like `Oasis` and `NFD` improved on this by using a sliding window of recent frames (e.g., the last 16 frames). This provides temporal consistency but is fundamentally short-term.
3.  **State-Space Models:** Models like `LSVM` (Po et al., 2025) try to compress the entire history into a compact latent state. This is efficient but the memory scope is still practically limited by the training sequence length.
4.  **Explicit Spatial Memory Models:** Models like `WorldMem` introduced an external memory bank of past frames, retrieving them based on spatial cues (camera pose). This enables true long-term memory but introduces new problems: retrieval inefficiency and a potential degradation of generative quality in new scenes.
5.  **Balanced Spatio-Temporal Memory (This Paper):** `Memory Forcing` represents the next step. It doesn't just add spatial memory; it actively teaches the model *how* and *when* to use it, balancing it with temporal memory. It also introduces a more advanced, geometry-based retrieval mechanism that is far more efficient and scalable.

## 3.4. Differentiation Analysis
`Memory Forcing` differentiates itself from prior work in two key aspects: **training strategy** and **memory retrieval mechanism**.

*   **vs. Temporal-Only Models (`Oasis`, `NFD`):** The primary difference is the explicit inclusion of a long-term spatial memory. While `NFD` excels at new scene generation, it fails at consistency on revisits, a problem `Memory Forcing` is designed to solve.
*   **vs. Pose-Based Spatial Memory (`WorldMem`):**
    *   **Retrieval:** `WorldMem` retrieves frames by checking for camera pose overlap. This is computationally expensive (linear scan over all past frames) and can be brittle. `Memory Forcing` uses a 3D point cloud; it retrieves frames linked to the geometry currently visible, which is faster ($O(1)$ complexity) and more robust.
    *   **Training:** `WorldMem` is trained with a simpler spatial conditioning. `Memory Forcing` introduces `Hybrid Training` and `Chained Forward Training` to explicitly teach the model to balance spatial and temporal cues, addressing `WorldMem`'s weakness in new scene generation.
*   **vs. State-Space Models (`LSVM`):** `LSVM` compresses history into a latent state, which is implicit and not easily interpretable. `Memory Forcing` uses an explicit 3D cache, making it clear what spatial evidence the model is using. This also allows for theoretically infinite-horizon memory, whereas state-space models are limited by their training context.

# 4. Methodology

## 4.1. Principles
The core principle of `Memory Forcing` is to resolve the trade-off between **generative quality in new environments** and **spatial consistency when revisiting old ones**. The authors observe that models tend to specialize: those good at one task are often poor at the other. The intuition behind `Memory Forcing` is that a model can be taught to be proficient at both if it learns to dynamically switch its dependency between two types of memory:
1.  **Temporal Memory (Recent Frames):** Best for generating smooth, continuous motion and exploring novel scenes where no prior spatial information exists.
2.  **Spatial Memory (Historical Frames):** Essential for reconstructing previously seen areas accurately and maintaining long-term consistency.

    The framework "forces" this adaptive behavior through two main avenues:
*   **Training Data & Protocol:** By exposing the model to distinct gameplay regimes (`Hybrid Training`) and simulating inference-time challenges (`Chained Forward Training`), it learns *when* to trust which memory source.
*   **Memory Architecture:** By providing a highly efficient and geometrically-grounded spatial memory (`Geometry-indexed Spatial Memory`), it ensures that retrieving long-term context is fast and reliable.

    The overall pipeline is illustrated in Figure 2.

    ![Figure 2: Memory Forcing Pipeline. Our framework combines spatial and temporal memory for video generation. 3D geometry is maintained through streaming reconstruction of key frames along the camera trajectory. During generation, Point-to-Frame Retrieval maps spatial context to historical frames, which are integrated with temporal memory and injected together via memory crossattention in the DiT backbone. Chained Forward Training creates larger pose variations, encouraging the model to effectively utilize spatial memory for maintaining long-term geometric consistency.](images/2.jpg)
    *该图像是示意图，展示了记忆强制框架在Minecraft中的应用，包括关键帧融合和相机轨迹的生成。图中展示了如何通过延长时序记忆和点到帧检索来维持空间一致性和生成质量。*

## 4.2. Core Methodology In-depth

### 4.2.1. Preliminaries: Autoregressive Video Diffusion
The method builds upon `Diffusion Forcing` for autoregressive video generation. A video is a sequence of frames $X^{1:T} = x_1, x_2, \ldots, x_T$. During training, each frame $x_t$ is corrupted with an independent noise level $k_t \in [0, 1]$, creating a noisy sequence $\tilde{X}^{1:T}$. The model, an epsilon-predicting model $\epsilon_\theta$, is trained to predict the noise added to all frames in the sequence. The objective function is a straightforward mean squared error loss between the true noise and the predicted noise:

\$
\mathcal { L } = \mathbb { E } _ { k ^ { 1 : T } , X ^ { 1 : T } , \epsilon ^ { 1 : T } } \left[ \vert \epsilon ^ { 1 : T } - \epsilon _ { \theta } ( \tilde { X } ^ { 1 : T } , k ^ { 1 : T } ) \vert ^ { 2 } \right]
\$

*   $\epsilon^{1:T}$: The sequence of ground-truth noise vectors added to the frames.
*   $\epsilon_{\theta}(\cdot)$: The diffusion model (a DiT) which predicts the noise.
*   $\tilde{X}^{1:T}$: The sequence of noisy input frames.
*   $k^{1:T}$: The sequence of noise levels for each frame.

    For interactive generation in Minecraft, the model is also conditioned on player actions $\mathcal{A}^{1:T}$, so the prediction becomes $\epsilon_{\theta}(\tilde{X}^{1:T}, k^{1:T}, \mathcal{A}^{1:T})$.

### 4.2.2. Memory-Augmented Architecture
The core model is a Diffusion Transformer (`DiT`). To enable it to use long-term memory, the authors augment it with two key components:

1.  **Spatial Memory Extraction:** Relevant historical frames are retrieved from memory using the `Point-to-Frame Retrieval` mechanism (detailed in Section 4.2.4).
2.  **Memory Cross-Attention:** The retrieved spatial memory frames are integrated into the `DiT` backbone using cross-attention layers. In each block of the Transformer, the tokens of the current frame being generated act as **queries**, while the tokens from the retrieved historical frames act as **keys** and **values**. This allows the model to "look up" relevant visual information from the past to inform the current generation. The attention mechanism is formulated as:

    \$
    \mathrm { A t t e n t i o n } ( \tilde { Q } , \tilde { K } _ { \mathrm { s p a t i a l } } , V _ { \mathrm { s p a t i a l } } ) = \mathrm { S o f t m a x } \left( \frac { \tilde { Q } \tilde { K } _ { \mathrm { s p a t i a l } } ^ { T } } { \sqrt { d } } \right) V _ { \mathrm { s p a t i a l } }
    \$

    *   $\tilde{Q}$: Queries from the current frame's tokens.
    *   $\tilde{K}_{\mathrm{spatial}}$: Keys from the retrieved spatial memory frames' tokens.
    *   $V_{\mathrm{spatial}}$: Values from the retrieved spatial memory frames' tokens.
    *   $d$: The dimension of the keys.

        Crucially, the queries $\tilde{Q}$ and keys $\tilde{K}_{\mathrm{spatial}}$ are augmented with **Plücker coordinates**, which encode the relative camera pose (position and orientation) between the current view and the historical views. This gives the attention mechanism explicit geometric information about how the views relate to each other in 3D space.

### 4.2.3. Autoregressive Diffusion Training with Memory Forcing

This is the central innovation of the paper. It consists of two training strategies.

#### **Hybrid Training**
The model is trained on a mix of data with different memory strategies. The context window of size $L$ is split into two halves. The first half ($L/2$ frames) is always the most recent temporal context. The second half is dynamically assigned based on the data source:

\$
{ \mathcal { W } } = [ { \mathcal { T } } _ { \mathrm { f i x e d } } , { \mathcal { M } } _ { \mathrm { c o n t e x t } } ] = \left\{ { \begin{array} { l l } { [ { \mathcal { T } } _ { \mathrm { f i x e d } } , { \mathcal { M } } _ { \mathrm { s p a t i a l } } ] } & \text{for revisit-heavy data (MineDojo)} \\ { [ { \mathcal { T } } _ { \mathrm { f i x e d } } , { \mathcal { T } } _ { \mathrm { e x t e n d e d } } ] } & \text{for exploration-heavy data (VPT)} \end{array} } \right.
\$

*   $\mathcal{W}$: The complete context window.
*   $\mathcal{T}_{\mathrm{fixed}}$: The fixed $L/2$ most recent frames (short-term temporal memory).
*   $\mathcal{M}_{\mathrm{context}}$: The dynamically assigned context.
*   $\mathcal{M}_{\mathrm{spatial}}$: Long-term spatial memory frames retrieved via the geometry-indexed method.
*   $\mathcal{T}_{\mathrm{extended}}$: Additional temporal frames from earlier in the sequence (long-term temporal memory).

    By applying the spatial memory strategy on the synthetic `MineDojo` dataset (which has frequent revisits) and the temporal-only strategy on the `VPT` dataset (which is more exploratory), the model learns to associate the presence of relevant spatial memory with revisit scenarios and to rely on temporal context otherwise.

#### **Chained Forward Training (CFT)**
`CFT` is designed to make the model more robust to the errors that accumulate during long autoregressive rollouts. Instead of always training on ground-truth context, `CFT` progressively replaces parts of the context with the model's own (quickly generated) predictions. The process is detailed in `Algorithm 1`.

The algorithm is as follows:
```
Algorithm 1 Chained Forward Training (CFT)

Require: Video x, conditioning inputs C, forward steps T, window size W, model ε_θ
1: Initialize F_pred ← ∅, L_total ← 0
2: for j = 0 to T - 1 do
3:   Construct window W_j:
4:   for k ∈ [j, j + W - 1] do
5:     if k ∈ F_pred then
6:       W_j[k - j] ← F_pred[k]  // Use predicted frame
7:     else
8:       W_j[k - j] ← x_k         // Use ground truth frame
9:     end if
10:  end for
11:  Compute L_j ← ||ε - ε_θ(W_j, C_j, t)||^2, update L_total ← L_total + L_j
12:  if j < T - 1 then
13:    x̂_{j+W-1} ← denoise(W_j, C_j) // Generate with fewer steps, no gradients
14:    F_pred[j + W - 1] ← x̂_{j+W-1} // Store for next window
15:  end if
16: end for
17: return L_chain ← L_total / T
```

**Step-by-step explanation:**
1.  The training iterates through a long video sequence, processing it in sliding windows of size $W$.
2.  In each step $j$, it constructs a context window $W_j$. For any frame in this window that was predicted in a previous step, it uses the **predicted frame** `F_pred[k]`. Otherwise, it uses the **ground-truth frame** $x_k$.
3.  The model's loss $L_j$ is computed based on this mixed ground-truth/predicted context.
4.  Crucially, before moving to the next window $j+1$, the model generates a quick prediction $x̂_{j+W-1}$ for the last frame of the current window. This prediction is stored and will be used as context in subsequent windows.
5.  This chaining of predictions creates larger pose variations and error propagation than standard training. This forces the model to learn to rely on the stable, external `spatial memory` (`M_spatial` inside $C_j$) to correct inconsistencies, rather than trusting the potentially drifting temporal context.

    The final training objective for `CFT` is the average loss over all chained windows:

\$
\mathcal { L } _ { \mathrm { c h a i n } } = \frac { 1 } { T } \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } _ { t , \epsilon } \left[ \| \epsilon - \epsilon _ { \theta } ( \mathcal { W } _ { j } ( \mathbf { x } , \hat { \mathbf { x } } ) , \mathcal { C } _ { j } , t ) \| ^ { 2 } \right] , \quad t \sim \mathrm { U n i f o r m } ( 0 , T _ { \mathrm { n o i s e } } ) , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )
\$

*   $\mathcal{W}_j(\mathbf{x}, \hat{\mathbf{x}})$: The context window at step $j$, containing a mix of ground-truth frames $\mathbf{x}$ and predicted frames $\hat{\mathbf{x}}$.
*   $\mathcal{C}_j$: Conditioning inputs for the window, including actions $A_j$, poses $\mathcal{P}_j$, and the retrieved spatial memory $\mathcal{M}_{\mathrm{spatial}}$.

### 4.2.4. Geometry-indexed Spatial Memory
This is the mechanism for efficiently storing and retrieving long-term spatial memory. It consists of two components that work together.

#### **Incremental 3D Reconstruction**
Instead of storing every frame, the system builds an explicit 3D representation of the scene (a global point cloud) in a streaming fashion.
*   **Keyframe Selection:** A new frame is designated a "keyframe" and used to update the 3D model only if it provides significant new information. A frame is a keyframe if it meets either condition:
    \$
    \mathrm { I s K e y f r a m e } ( t ) = \mathrm { N o v e l C o v e r a g e } ( I _ { t } , \mathcal { G } _ { \mathrm { g l o b a l } } ) \ \mathbf { o r } \ ( | \mathcal { H } _ { t } | < \tau _ { \mathrm { h i s t } } )
    \$
    *   `NovelCoverage(...)`: A function that checks if the current frame $I_t$ sees a sufficient amount of new area not covered by the existing global geometry $\mathcal{G}_{\mathrm{global}}$.
    *   $|\mathcal{H}_t| < \tau_{\mathrm{hist}}$: A condition that checks if there are too few historical frames available for the current view (here, $\tau_{\mathrm{hist}} = L/2=8$). This ensures there's always some context.
*   **Reconstruction:** When a keyframe is added, the system uses the `VGGT` model to predict its depth map. This depth map is then back-projected into a 3D point cloud using the camera's extrinsic matrix $E$:
    \$
    { \bf E } = \left[ \begin{array} { c c } { { \bf R } ( pitch , yaw ) } & { - { \bf R C } } \\ { { \bf 0 } ^ { T } } & { 1 } \end{array} \right]
    \$
    *   $\mathbf{R}(pitch, yaw)$: The rotation matrix derived from the camera's orientation.
    *   $\mathbf{C} = [x, y, z]^T$: The camera's position in 3D space.
*   **Scale Alignment & Integration:** To ensure depth maps from different windows are consistent, a `Cross-Window Scale Alignment` module (detailed in Appendix A.1) aligns the scale of the new depth maps with the existing global geometry. The new 3D points are then integrated into the global point cloud using voxel downsampling to keep the point density bounded, ensuring memory usage scales with spatial area, not time.

#### **Point-to-Frame Retrieval**
This is the retrieval mechanism. For any given moment in the game, it efficiently finds the most relevant historical frames.
1.  The global 3D point cloud is projected onto the current camera view.
2.  Each point in the cloud has a tag indicating which source frame it originally came from.
3.  The system counts which source frames are most frequently seen among the visible points.
4.  The top-k (here, k=8) most frequent source frames are selected as the spatial memory context $\mathcal{H}_t$.

    The retrieval process is formalized as:
\$
\mathcal { H } _ { t } = \arg \operatorname* { m a x } _ { k = 1 , \ldots , 8 } \mathrm { C o u n t } ( \operatorname { s o u r c e } ( p _ { i } ) : p _ { i } \in \mathcal { P } _ { \mathrm { v i s i b l e } } ^ { t } )
\$

*   $\mathcal{P}_{\mathrm{visible}}^t$: The set of global 3D points visible in the current frame $t$.
*   $\mathrm{source}(p_i)$: A function that returns the index of the source frame from which point $p_i$ was created.
*   $\mathcal{H}_t$: The set of top-8 historical frames.

    This geometry-based retrieval is very efficient. Because the point cloud is downsampled, the number of points to check is bounded, giving the retrieval a constant time complexity, $O(1)$, regardless of how long the game has been played.

# 5. Experimental Setup

## 5.1. Datasets
The experiments use a combination of existing and newly constructed datasets to evaluate the model from multiple angles.

*   **Training Datasets:**
    1.  **VPT (Video PreTraining)**: A large-scale dataset (over 2000 hours originally, filtered down) of human gameplay in Minecraft. This data is exploration-heavy and is used for the "temporal-only" part of the `Hybrid Training`.
    2.  **Synthetic MineDojo Dataset**: A dataset of 11,000 videos generated using the `MineDojo` simulator. Each video contains a 1,500-frame sequence where the agent frequently revisits locations. This data is revisit-heavy and is used for the "spatio-temporal" part of `Hybrid Training`.

*   **Evaluation Datasets:** All constructed using `MineDojo`.
    1.  **Long-term Memory**: 150 long video sequences (1,500 frames each) with many revisits, taken from the `WorldMem` dataset, to test spatial consistency.
    2.  **Generalization Performance**: 150 video sequences (800 frames each) across nine Minecraft terrain types *not seen during training* (e.g., `extreme hills`, `swampland`, `mesa`). This tests the model's ability to adapt to new environments.
    3.  **Generation Performance**: 300 video sequences (800 frames each) in new environments to assess the general quality of generated scenes.

        The use of these specific datasets is effective because they directly correspond to the two core capabilities the paper aims to improve: long-term consistency (revisit-heavy data) and generalization/quality (unseen/new environments).

## 5.2. Evaluation Metrics
The paper uses four standard metrics to evaluate the quality of the generated videos.

### 5.2.1. Fréchet Video Distance (FVD)
*   **Conceptual Definition:** FVD measures the perceptual quality and temporal consistency of generated videos. It compares the distribution of features extracted from generated videos to the distribution of features from real videos. A lower FVD score indicates that the generated videos are more similar to real videos in terms of both appearance and motion. It is analogous to the Fréchet Inception Distance (FID) for images but extended to videos.
*   **Mathematical Formula:** FVD is calculated as the Wasserstein-2 distance between two multivariate Gaussian distributions fitted to the features extracted by a pre-trained video model (e.g., an I3D network).
    \$
    \text{FVD}(g, r) = \left\| \mu_g - \mu_r \right\|_2^2 + \text{Tr}\left(\Sigma_g + \Sigma_r - 2(\Sigma_g \Sigma_r)^{1/2}\right)
    \$
*   **Symbol Explanation:**
    *   $\mu_g$ and $\mu_r$: The mean feature vectors of the generated and real videos, respectively.
    *   $\Sigma_g$ and $\Sigma_r$: The covariance matrices of the features of the generated and real videos, respectively.
    *   $\text{Tr}(\cdot)$: The trace of a matrix.
*   **Goal:** **Lower is better.**

### 5.2.2. Learned Perceptual Image Patch Similarity (LPIPS)
*   **Conceptual Definition:** LPIPS measures the perceptual similarity between two images. Unlike pixel-wise metrics like PSNR or SSIM, LPIPS uses features from deep neural networks (e.g., VGG, AlexNet) that are trained to be more aligned with human perception of similarity. It compares the feature activations between a generated frame and its ground-truth counterpart.
*   **Mathematical Formula:**
    \$
    d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l) \right\|_2^2
    \$
*   **Symbol Explanation:**
    *   $x, x_0$: The two images being compared (generated and ground truth).
    *   $l$: Index of a layer in the deep network.
    *   $\hat{y}^l, \hat{y}_0^l$: Feature activations from layer $l$ for images $x$ and $x_0$.
    *   $H_l, W_l$: Spatial dimensions of the feature map.
    *   $w_l$: A scaling factor for channel activations at layer $l$.
*   **Goal:** **Lower is better.**

### 5.2.3. Peak Signal-to-Noise Ratio (PSNR)
*   **Conceptual Definition:** PSNR measures the pixel-level reconstruction quality of an image by comparing the maximum possible pixel value to the amount of noise (error) present. It is a widely used metric for image compression and restoration quality, but it does not always correlate well with human perception.
*   **Mathematical Formula:**
    \$
    \text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)
    \$
    where MSE is the Mean Squared Error: `\text{MSE} = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j) - K(i,j)]^2`.
*   **Symbol Explanation:**
    *   $\text{MAX}_I$: The maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image).
    *   `I, K`: The ground-truth and generated images.
    *   `m, n`: The dimensions of the images.
*   **Goal:** **Higher is better.**

### 5.2.4. Structural Similarity Index Measure (SSIM)
*   **Conceptual Definition:** SSIM measures the structural similarity between two images, considering luminance, contrast, and structure. It is designed to be more consistent with human visual perception of quality than PSNR.
*   **Mathematical Formula:**
    \$
    \text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
    \$
*   **Symbol Explanation:**
    *   $\mu_x, \mu_y$: The mean of images $x$ and $y$.
    *   $\sigma_x^2, \sigma_y^2$: The variance of images $x$ and $y$.
    *   $\sigma_{xy}$: The covariance of $x$ and $y$.
    *   $c_1, c_2$: Small constants to stabilize the division.
*   **Goal:** **Higher is better.**

## 5.3. Baselines
The paper compares its method against three representative baseline models:
*   **Oasis (Decart et al., 2024):** A state-of-the-art interactive world model that relies on temporal context. Represents the "temporal-only" approach.
*   **NFD (Cheng et al., 2025):** Another strong temporal-only diffusion model for Minecraft, known for its high-speed generation.
*   **WorldMem (Xiao et al., 2025):** The primary long-term memory baseline. It uses pose-based retrieval to achieve spatial consistency. This comparison is crucial to show the benefits of `Memory Forcing`'s training strategies and efficient geometry-based retrieval.

    For fairness, all models are configured with a 16-frame context window and trained on identical datasets to ensure a consistent evaluation.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The paper's core results, presented in Table 1, demonstrate that `Memory Forcing` consistently outperforms all baseline models across three distinct evaluation settings. The evaluation is performed on frames 600-800 of long sequences, specifically testing long-horizon capabilities.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">Long-term Memory</th>
<th colspan="4">Generalization Performance</th>
<th colspan="4">Generation Performance</th>
</tr>
<tr>
<th>FVD ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>FVD ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>FVD ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Oasis</td>
<td>196.8</td>
<td>16.83</td>
<td>0.5654</td>
<td>0.3791</td>
<td>477.3</td>
<td>14.74</td>
<td>0.5175</td>
<td>0.5122</td>
<td>285.7</td>
<td>14.51</td>
<td>0.5063</td>
<td>0.4704</td>
</tr>
<tr>
<td>NFD</td>
<td>220.8</td>
<td>16.35</td>
<td>0.5819</td>
<td>0.3891</td>
<td>442.6</td>
<td>15.49</td>
<td>0.5564</td>
<td>0.4638</td>
<td>349.6</td>
<td>14.64</td>
<td>0.5417</td>
<td>0.4343</td>
</tr>
<tr>
<td>WorldMem</td>
<td>122.2</td>
<td>19.32</td>
<td>0.5983</td>
<td>0.2769</td>
<td>328.3</td>
<td>16.23</td>
<td>0.5178</td>
<td>0.4336</td>
<td>290.8</td>
<td>14.71</td>
<td>0.4906</td>
<td>0.4531</td>
</tr>
<tr>
<td>Ours</td>
<td><strong>84.9</strong></td>
<td><strong>21.41</strong></td>
<td><strong>0.6692</strong></td>
<td><strong>0.2156</strong></td>
<td><strong>253.7</strong></td>
<td><strong>19.86</strong></td>
<td><strong>0.6341</strong></td>
<td><strong>0.2896</strong></td>
<td><strong>185.9</strong></td>
<td><strong>17.99</strong></td>
<td><strong>0.6155</strong></td>
<td><strong>0.3031</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Long-term Memory:** In this setting, designed to test consistency during revisits, the proposed method (`Ours`) achieves a significantly lower FVD (84.9) and LPIPS (0.2156), and higher PSNR/SSIM compared to all others. It dramatically outperforms the temporal-only models `Oasis` and `NFD`. More importantly, it is substantially better than `WorldMem` (FVD 122.2), indicating that the combination of `CFT` and geometry-based retrieval leads to more accurate and stable consistency. This is qualitatively supported by Figure 3, where `Ours` generates a scene almost identical to the ground truth upon revisit, while `WorldMem` shows artifacts and `NFD` generates a completely different scene.

    ![Figure 3: Memory capability comparison across different models for maintaining spatial consistency and scene coherence when revisiting previously observed areas.](images/3.jpg)
    *该图像是插图，展示了不同模型在Minecraft场景生成上的表现，包括GT、NFD、WorldMen和Ours。通过比较这些模型在多个时间步的输出，可以观察到生成内容的自然性和空间一致性。*

*   **Generalization Performance:** On unseen terrains, `Memory Forcing` again leads by a large margin. The FVD score of 253.7 is much lower than `WorldMem`'s 328.3 and the even higher scores of `Oasis` and `NFD`. This result is critical because it validates the effectiveness of the `Hybrid Training` protocol. By learning to rely on temporal memory during exploration, the model avoids the degradation in new scene quality that affects `WorldMem`, which over-relies on its spatial memory mechanism. Figure 4 (top) visually confirms this, showing stable and high-quality generation on new terrain, while baselines show artifacts or inconsistencies.

*   **Generation Performance:** This setting assesses overall quality in new environments. Once again, `Memory Forcing` achieves the best scores across all metrics. This demonstrates that the framework successfully resolves the central trade-off: it enhances long-term memory without sacrificing, and in fact improving, the model's core generative capability. Figure 4 (bottom) highlights this, showing that `Ours` produces dynamic and realistic scenes where distant objects become clearer as the player approaches, a behavior other models fail to capture correctly.

    ![Figure 4: Generalization performance on unseen terrain types (top) and generation performance in new environments (bottom). Our method demonstrates superior visual quality and responsive movement dynamics, with distant scenes progressively becoming clearer as the agent approaches, while baselines show quality degradation, minimal distance variation, or oversimplified distant scenes.](images/4.jpg)
    *该图像是插图，展示了在Minecraft中不同方法生成场景的对比，包括Oasis、NFD、WorldMem和我们的模型在不同时间步（600和800处）的效果。图中显示了四种方法在生成场景一致性和质量方面的差异。*

## 6.2. Data Presentation (Tables)

### 6.2.1. Efficiency of Geometry-Indexed Spatial Memory (Table 2)
This table compares the retrieval speed (Frames Per Second, FPS) and memory usage of the proposed `Geometry-indexed Spatial Memory` against `WorldMem`'s pose-based retrieval.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th>Frame Range</th>
<th colspan="2">0-999</th>
<th colspan="2">1000-1999</th>
<th colspan="2">2000-2999</th>
<th colspan="2">3000-3999</th>
<th colspan="2">Total (0-3999)</th>
</tr>
<tr>
<th>Method</th>
<th>Speed (FPS ↑)</th>
<th>Mem. (Count ↓)</th>
<th>Speed (FPS ↑)</th>
<th>Mem. (Count ↓)</th>
<th>Speed (FPS ↑)</th>
<th>Mem. (Count ↓)</th>
<th>Speed (FPS ↑)</th>
<th>Mem. (Count ↓)</th>
<th>Speed (FPS ↑)</th>
<th>Mem. (Count ↓)</th>
</tr>
</thead>
<tbody>
<tr>
<td>WorldMem</td>
<td>10.11</td>
<td>+1000</td>
<td>3.43</td>
<td>+1000</td>
<td>2.06</td>
<td>+1000</td>
<td>1.47</td>
<td>+1000</td>
<td>4.27</td>
<td>4000</td>
</tr>
<tr>
<td>Ours</td>
<td>18.57</td>
<td>+25.45</td>
<td>27.08</td>
<td>+19.70</td>
<td>41.36</td>
<td>+14.55</td>
<td>37.84</td>
<td>+12.95</td>
<td><strong>31.21</strong></td>
<td><strong>72.65</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
The results are striking. `WorldMem`'s retrieval speed plummets as the sequence length increases (from 10.11 FPS to 1.47 FPS), because its linear search complexity scales with the size of the memory bank, which grows by 1000 frames in each segment. In contrast, the proposed method (`Ours`) maintains a high and even increasing retrieval speed. This is because its retrieval complexity is constant ($O(1)$) and its memory storage scales with spatial coverage, not temporal duration. By the end of the 4000-frame sequence, `Ours` uses 98.2% less memory (72.65 frames vs. 4000) and is 7.3 times faster overall (31.21 FPS vs. 4.27 FPS). This confirms the superior efficiency and scalability of the `Geometry-indexed Spatial Memory`.

## 6.3. Ablation Studies / Parameter Analysis
Table 3 presents ablation studies that dissect the contributions of the different components of `Memory Forcing`.

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th colspan="2">Training Strategies</th>
<th colspan="2">Retrieval Strategies</th>
<th colspan="4">Metrics</th>
</tr>
<tr>
<th>HT-w/o-CFT</th>
<th>MF</th>
<th>Pose-based</th>
<th>3D-based</th>
<th>FVD ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>✓</td>
<td></td>
<td></td>
<td>;</td>
<td>366.1</td>
<td>15.09</td>
<td>0.5649</td>
<td>0.4122</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td>230.4</td>
<td>16.24</td>
<td>0.5789</td>
<td>0.3598</td>
</tr>
<tr>
<td>✓</td>
<td></td>
<td>;</td>
<td></td>
<td>225.9</td>
<td>16.24</td>
<td>0.5945</td>
<td>0.3722</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td>✓</td>
<td><strong>165.9</strong></td>
<td><strong>18.17</strong></td>
<td><strong>0.6222</strong></td>
<td><strong>0.2876</strong></td>
</tr>
</tbody>
</table>

*(Note: The table in the paper seems to have formatting issues and missing labels for the rows. Based on the context, the analysis is as follows, assuming the rows represent different combinations of training and retrieval strategies.)*

**Training Strategy Analysis:**
*   **Fine-Tuning (FT) vs. Hybrid Training (HT-w/o-CFT):** Comparing a simple fine-tuning approach (assumed to be the baseline for comparison) with `Hybrid Training` shows a significant improvement (e.g., FVD drops from ~230.4 to 225.9, though the table is slightly ambiguous). The paper states HT improves performance by integrating real and synthetic data, teaching the model to start balancing memory sources.
*   **Hybrid Training vs. Full Memory Forcing (MF):** The full `Memory Forcing` strategy, which includes `Chained Forward Training (CFT)`, yields the best results (FVD 165.9). This demonstrates that `CFT` is crucial. By exposing the model to its own cascading errors during training, it forces a stronger reliance on the stable spatial memory, leading to optimal performance in balancing generation and consistency.

**Retrieval Mechanism Comparison:**
*   **Pose-based vs. 3D-based Retrieval:** Comparing a model using pose-based retrieval (like `WorldMem`) with one using the proposed 3D-based `Geometry-indexed Spatial Memory` shows a massive performance gain. The FVD drops from 225.9 (with pose-based) to 165.9 (with 3D-based). This confirms that the geometry-anchored retrieval is not only more efficient (as per Table 2) but also more effective, providing more relevant and accurate historical context for the model to use.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully identifies and addresses a critical trade-off in generative world models: the conflict between generating novel scenes and maintaining long-term spatial consistency. The proposed framework, **`Memory Forcing`**, provides a comprehensive solution through two key innovations. First, its specialized training protocols (`Hybrid Training` and `Chained Forward Training`) effectively teach the model to dynamically balance its reliance on temporal and spatial memory based on the context (exploration vs. revisit). Second, its `Geometry-indexed Spatial Memory` system offers a highly efficient and scalable method for managing long-term history, using 3D reconstruction and point-to-frame mapping to achieve constant-time retrieval. The extensive experiments demonstrate that `Memory Forcing` sets a new state-of-the-art in Minecraft scene generation, achieving superior performance in long-term consistency, generalization to new environments, and overall generative quality, all while being significantly more computationally efficient than previous spatial memory approaches.

## 7.2. Limitations & Future Work
The authors acknowledge two primary limitations:
1.  **Domain Specificity:** The current implementation is heavily validated on Minecraft. Its direct applicability to other gaming environments or real-world video generation is not guaranteed and would likely require domain-specific adaptations.
2.  **Fixed Resolution:** The model operates at a relatively low resolution of 384x224 pixels. While sufficient for many gameplay scenarios, this may not be adequate for applications demanding high-fidelity visuals.

    For future work, the authors plan to:
1.  **Extend to Diverse Environments:** Adapt the framework to work in other games and real-world scenarios, which would involve developing techniques to handle different visual styles and dynamics.
2.  **Improve Resolution and Efficiency:** Explore higher-resolution generation and integrate advanced acceleration techniques to enhance performance for real-time interactive use cases.

## 7.3. Personal Insights & Critique
This paper presents a very well-thought-out and elegant solution to a tangible problem in long-form video generation. My key takeaways and critiques are:

**Positive Insights:**
*   **The "Teaching" Metaphor:** The concept of `Memory Forcing` is powerful because it frames the training process as "teaching" the model a complex, adaptive behavior. Instead of just providing a new architectural component (spatial memory), the authors designed a curriculum (`Hybrid Training` + `CFT`) to ensure the model learns to use it correctly. This is a sophisticated approach to model training that goes beyond simple optimization.
*   **Efficiency as a First-Class Citizen:** The design of the `Geometry-indexed Spatial Memory` is a standout contribution. In an era where models are becoming computationally immense, designing a system that offers theoretically constant-time retrieval and memory that scales with space rather than time is both clever and practical. This makes the approach viable for truly long-running, "infinite-horizon" applications.
*   **Synergy of Components:** The strength of `Memory Forcing` lies in the tight integration of its parts. The `CFT` strategy would be less effective without a reliable and fast spatial memory to fall back on. Conversely, the advanced memory system's benefits are fully realized because the training protocols teach the model how to leverage it optimally.

**Potential Issues and Critique:**
*   **Dependency on Reconstruction Quality:** The entire memory system hinges on the performance of the underlying 3D reconstruction model (`VGGT`). If the depth prediction or pose estimation from `VGGT` is inaccurate, the global point cloud will be corrupted. This could lead to incorrect frame retrieval, potentially harming consistency more than helping it. The paper doesn't deeply analyze the system's robustness to reconstruction errors.
*   **System Complexity:** The overall system is quite complex, involving a `DiT` for generation, a separate `VGGT` model for geometry, a global point cloud manager, and multiple specialized training loops. This complexity could make it difficult to implement, debug, and scale. The practical engineering challenges might be significant.
*   **Applicability to Non-Rigid Scenes:** The geometry-based approach works well for Minecraft, where the world is largely static and blocky. It is less clear how this method would perform in environments with many dynamic, non-rigid objects (e.g., crowds of characters, flowing water, deformable terrain). The point-to-frame mapping might become less reliable in such scenarios.

    Overall, `Memory Forcing` is a strong piece of research that pushes the boundaries of what's possible in generative world modeling. Its core ideas—teaching models adaptive behaviors and designing highly efficient, geometrically grounded memory systems—are likely to be influential in the development of future AI agents for complex, interactive environments.