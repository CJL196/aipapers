# 1. Bibliographic Information

## 1.1. Title
Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval

The title clearly outlines the paper's core concepts:
1.  **Core Problem:** Achieving scene consistency in interactive long video generation.
2.  **Proposed Solution:** A method named `Context-as-Memory`.
3.  **Key Mechanism:** Utilizing a `Memory Retrieval` module.

## 1.2. Authors
The authors are Jiwen Yu, Yiran Qin, and Xihui Lu from The University of Hong Kong; Jianhong Bai from Zhejiang University; and Quande Liu, Xintao Wang, Pengfei Wan, and Di Zhang from the Kling Team at Kuaishou Technology. The collaboration between academia and a major tech company (Kuaishou, known for its short-video platform) suggests a blend of rigorous academic research and industry-level application focus, particularly in the domain of video generation.

## 1.3. Journal/Conference
The paper is available as a preprint on arXiv. arXiv is a widely used open-access repository for academic papers in fields like physics, mathematics, computer science, and quantitative biology. While not a peer-reviewed publication venue itself, it is a standard platform for researchers to share their findings early, often before or during the peer-review process for a major conference or journal. This indicates the work is recent and represents the current state of research.

## 1.4. Publication Year
The paper was submitted to arXiv with a publication date of June 3, 2025.

## 1.5. Abstract
The abstract identifies a key limitation in existing interactive video generation models: their struggle with maintaining scene consistency in long videos due to insufficient use of historical context. To address this, the authors propose `Context-as-Memory`, a method that uses past frames as a memory bank. The approach features two simple designs: (1) storing context as raw frames without complex processing, and (2) conditioning the generation model by simply concatenating these context frames with the frames to be predicted. To manage the computational cost of using all historical frames, they introduce a `Memory Retrieval` module. This module intelligently selects the most relevant context frames by identifying Field of View (FOV) overlap based on camera poses. Experiments show that this method significantly outperforms state-of-the-art (SOTA) models in memory capability and can even generalize to open-domain scenarios not seen during training.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2506.03141
*   **PDF Link:** https://arxiv.org/pdf/2506.03141v2
*   **Publication Status:** This is a preprint and has not yet been published in a peer-reviewed venue.

# 2. Executive Summary

## 2.1. Background & Motivation
*   **Core Problem:** The primary challenge addressed is the lack of **long-term memory** in video generation models. When generating long, interactive videos (e.g., for games or simulations), models often fail to maintain scene consistency. For example, if a user controls the camera to turn away from an object and then turn back, an ideal model should regenerate the same object and scene. However, current models often generate a completely new scene, breaking the illusion of a persistent, coherent world.

*   **Importance & Gaps:** This problem is a major roadblock for applications that rely on creating immersive and believable virtual worlds, such as video games, virtual reality, and simulators for autonomous systems. Existing approaches have significant limitations:
    *   **Limited Context Window:** Methods like `Diffusion Forcing` only use a fixed window of the most recent frames (e.g., a few dozen). This provides short-term continuity but fails to recall information from the distant past.
    *   **Explicit 3D Reconstruction:** Some works attempt to build a 3D model of the scene from generated frames. This provides a strong memory but is computationally expensive, slow, and prone to accumulating errors over time, making it impractical for continuous, long-form generation.
    *   **Information Loss:** Other methods like `FramePack` compress the entire history, but this often leads to significant information loss, especially for older frames.

*   **Innovative Idea:** The paper's central idea is both simple and powerful: **treat the historical context of generated frames as the memory itself**. Instead of converting frames into another representation (like 3D models or compressed features), the model directly uses a selection of past frames as a visual reference to generate new ones. The innovation lies not just in using context, but in proposing an efficient method (`Memory Retrieval`) to select the *most relevant* pieces of context based on camera geometry, thereby making the approach computationally feasible.

## 2.2. Main Contributions / Findings
The paper makes the following key contributions:
1.  **A Novel Framework (`Context-as-Memory`):** It proposes a framework for scene-consistent long video generation that directly utilizes past frames as memory. The conditioning mechanism is remarkably simple: it concatenates the selected memory frames with the frames to be predicted at the model's input, requiring no complex external modules like adapters.

2.  **An Efficient Retrieval Module (`Memory Retrieval`):** To overcome the computational burden of using all past frames, the paper introduces a rule-based `Memory Retrieval` module. This module leverages camera trajectory information to calculate the Field of View (FOV) overlap between past frames and the future frame, allowing it to retrieve only the frames that show the same parts of the scene.

3.  **A New High-Quality Dataset:** The authors collected a new dataset for this task using Unreal Engine 5. The dataset consists of long videos (over 7,600 frames each) with diverse scenes and, crucially, precise camera pose annotations for every frame. This data is essential for training and evaluating models on long-term consistency.

4.  **Superior Performance:** Experimental results demonstrate that `Context-as-Memory` significantly outperforms existing state-of-the-art methods in maintaining scene consistency. The model not only achieves better quantitative scores but also shows strong generalization to open-domain scenes that were not part of its training data.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must be familiar with the following concepts:

*   **Diffusion Models:** These are a class of generative models that learn to create data by reversing a gradual noising process.
    1.  **Forward Process (Noising):** Start with a real data sample (e.g., an image or video latent) and slowly add Gaussian noise over a series of timesteps $T$. After $T$ steps, the data becomes pure noise.
    2.  **Reverse Process (Denoising):** A neural network is trained to reverse this process. At each timestep $t$, it takes the noisy data $\mathbf{z}_t$ and predicts the noise $\epsilon$ that was added. By subtracting this predicted noise, it can gradually recover the original clean data from pure noise. The model is trained to minimize the difference between the actual noise and the predicted noise.

*   **Latent Diffusion Models (LDMs):** Training diffusion models directly on high-resolution images or videos is computationally very expensive. LDMs solve this by first using an autoencoder to compress the data into a smaller, lower-dimensional **latent space**.
    *   **Encoder:** A neural network that maps a high-resolution video $\mathbf{x}$ to a low-dimensional latent representation $\mathbf{z}$.
    *   **Diffusion Process:** The noising and denoising process happens entirely within this efficient latent space.
    *   **Decoder:** A neural network that maps the final denoised latent $\mathbf{z}_0$ back to a high-resolution video $\mathbf{x}$.

*   **Diffusion Transformer (DiT):** The neural network used for the denoising step in diffusion models was traditionally a U-Net architecture. DiT, proposed by Peebles and Xie (2023), replaced the U-Net with a **Transformer**. Transformers are known for their scalability and strong performance in processing sequences of tokens, making them well-suited for modeling dependencies in the latent space. In this paper, the DiT processes a sequence of latent video frames.

*   **Rotary Position Embedding (RoPE):** Transformers need a way to understand the order of tokens in a sequence. RoPE is a type of positional encoding that encodes position information by rotating the feature vectors based on their absolute position. A key advantage of RoPE is its ability to handle sequences of variable lengths naturally, which is crucial for this paper's method of concatenating a variable number of context frames.

## 3.2. Previous Works
The paper positions itself relative to several key areas of research:

*   **Streaming Video Generation:** This is the task of generating video continuously, frame by frame or segment by segment, where each new segment is conditioned on the previously generated ones.
    *   **`Diffusion Forcing Transformer (DFoT)` [Song et al. 2025]:** A representative work in this area. It conditions the generation of the next video segment on a fixed-size window of the most recently generated frames. Its primary limitation is its short-term memory; it cannot recall scenes from further back in the video's history.
    *   **`FramePack` [Zhang and Agrawala 2025]:** This method attempts to use the entire history by hierarchically compressing past frames into a fixed number of context frames. However, it uses an exponential decay scheme where older frames are compressed much more aggressively, leading to significant information loss and a weak long-term memory.

*   **Memory-Enhanced Video Generation:**
    *   **3D Reconstruction Methods:** These approaches explicitly build a 3D representation (like a NeRF or mesh) of the scene from the generated video. To generate new frames, they can render a view from this 3D model to ensure consistency. The paper argues this is too slow and prone to error accumulation for real-time, interactive generation.
    *   **`WorldMem` [Xiao et al. 2025]:** This work also aims to provide memory. It injects features from historical frames into the generation model using a `cross-attention` mechanism. The current paper suggests this was only validated on short videos (~10 seconds) and in limited scenarios (Minecraft).

*   **Controllable Video Generation:** This field focuses on allowing users to control aspects of the generated video, such as camera motion or character actions. This paper builds on camera control methods to enable interactive exploration and to obtain the camera poses needed for its `Memory Retrieval` module.

## 3.3. Technological Evolution
The field of video generation has evolved rapidly:
1.  **Short, Single-Shot Generation:** Early models focused on generating short, fixed-length video clips from a text prompt. The main challenge was achieving high visual quality and temporal coherence within that short clip.
2.  **Video Continuation:** The next step was to extend videos. Models were trained to take an existing clip and generate the next few seconds, creating slightly longer videos.
3.  **Long Video Generation:** To create videos of minutes or more, researchers developed streaming or autoregressive methods. Here, the primary challenge shifted from short-term coherence to **long-term consistency and memory**. Early attempts suffered from "content drift," where scenes would slowly and illogically transform over time.
4.  **Interactive Long Video Generation:** The current frontier involves not just generating long videos but allowing a user to control the generation process in real-time (e.g., by steering the camera). This makes the memory problem even more acute, as user interaction can create complex trajectories that revisit past locations, demanding perfect scene recall.

    This paper fits into the fourth stage, directly tackling the memory problem that is central to creating believable interactive experiences.

## 3.4. Differentiation Analysis
The core innovation of `Context-as-Memory` lies in its **simplicity and directness** compared to prior work:

*   **vs. `DFoT`:** `DFoT` uses a "blind," recency-based context (the last N frames). `Context-as-Memory` uses an "intelligent," relevance-based context selected from the *entire history* based on geometric cues (FOV overlap).
*   **vs. `FramePack`:** `FramePack` aggressively compresses history, losing information. `Context-as-Memory` uses uncompressed, raw frames, preserving full visual detail. It manages complexity not by compression, but by selective retrieval.
*   **vs. 3D Reconstruction:** 3D methods build an intermediate abstract representation. `Context-as-Memory` skips this step, using the frames directly as the memory representation, which is faster and avoids accumulated reconstruction errors.
*   **vs. `WorldMem`:** `WorldMem` requires modifying the model architecture with cross-attention modules to inject memory. `Context-as-Memory` requires no architectural changes, simply concatenating context at the input, making it easier to integrate with existing models.

# 4. Methodology

## 4.1. Principles
The fundamental principle of `Context-as-Memory` is that previously generated video frames are the most direct and information-rich form of memory for maintaining scene consistency. Instead of transforming this visual data into another format (e.g., feature embeddings, 3D models), the method leverages it directly. The main challenge then becomes how to select the most relevant historical frames and inject them into the generation process efficiently. The proposed solution involves a simple concatenation-based conditioning and an intelligent, geometry-aware retrieval mechanism.

## 4.2. Core Methodology In-depth

### 4.2.1. Preliminaries: Base Model and Camera Control

The method is built upon a standard text-to-video latent diffusion model.

**1. Full-Sequence Text-to-Video Base Model:**
The foundation is a **Latent Video Diffusion Model** that uses a **Diffusion Transformer (DiT)**.
*   A **3D VAE** (Variational Autoencoder) is used to handle video data. Its `Encoder` maps a sequence of video frames $\mathbf{x}$ into a compressed latent representation $\mathbf{z} = Encoder(\mathbf{x})$.
*   The core of the generative model is a **DiT**, which operates on this latent sequence $\mathbf{z}$. During training, a clean latent $\mathbf{z}_0$ is corrupted with Gaussian noise $\epsilon$ to produce a noisy latent $\mathbf{z}_t$ at timestep $t$. The DiT, denoted as $\epsilon_{\phi}$, is trained to predict the added noise $\epsilon$ from $\mathbf{z}_t$, a text prompt $\mathbf{p}$, and the timestep $t$.
*   The training objective is the standard diffusion loss:
    $$
    \mathcal{L}(\phi) = \mathbb{E}[||\epsilon_{\phi}(\mathbf{z}_t, \mathbf{p}, t) - \epsilon||]
    $$
    where $\phi$ are the model parameters.
*   During inference, the model starts with random noise and iteratively denoises it using the trained network $\epsilon_{\phi}$ to produce a clean latent $\mathbf{z}$. Finally, the VAE's `Decoder` maps this latent back to a video sequence: $\mathbf{x} = Decoder(\mathbf{z})$.

**2. Camera-Conditioned Video Generation:**
To make the generation interactive, the model is conditioned on camera motion.
*   A camera trajectory is represented by a sequence of poses $\mathbf{cam} = [R, t] \in \mathbb{R}^{f \times (3 \times 4)}$, where $f$ is the number of frames, $R$ is the rotation matrix, and $t$ is the translation vector.
*   This camera information is injected into the DiT. A small camera encoder $\mathcal{E}_c(\cdot)$ (a simple MLP) projects the camera poses to match the dimensionality of the model's internal features. This camera embedding is then added to the output of the spatial attention module within each DiT block before it enters the 3D (spatiotemporal) attention module.
    $$
    \mathbf{F}_i = \mathbf{F}_o + \mathcal{E}_c(\mathbf{cam})
    $$
    where $\mathbf{F}_o$ is the output feature from spatial attention and $\mathbf{F}_i$ is the input to 3D attention.
*   The diffusion loss is updated to include the camera condition:
    $$
    \mathcal{L}_{\mathbf{cam}}(\phi, \phi_{MLP}) = \mathbb{E}[||\epsilon_{\phi, \phi_{MLP}}(\mathbf{z}_t, \mathbf{p}, \mathbf{cam}, t) - \epsilon||]
    $$

### 4.2.2. Context Frames Learning Mechanism for Memory

This is the core of how memory is injected. The method is simple and elegant, requiring no new modules.

The model architecture for this mechanism is shown in Figure 2 from the original paper.![Fig. 2. Model Architecture. We concatenate the context to be conditioned and the predicted frames along the frame dimension. This method of injecting context is simple and effective, requiring no additional modules.](images/2.jpg)
*该图像是示意图，展示了基于历史上下文进行视频生成的模型架构。通过将历史上下文与当前输出进行拼接，模型利用多种注意力机制（包括3D 和 2D 注意力）对帧进行处理，以提高生成视频的内容一致性。*

*   **Input Concatenation:** Suppose the model needs to predict a new sequence of latents, which are currently noisy ($\mathbf{z}_t$). Let the selected historical context frames be represented by their clean latents, $\mathbf{z}^c$. The model conditions on this context by concatenating them along the frame (temporal) dimension. The new input to the DiT is the combined sequence $\{\mathbf{z}_t, \mathbf{z}^c\}$.
*   **Shared Attention:** Both the noisy latents to be predicted and the clean context latents are processed together by the self-attention layers of the DiT. This allows the model to learn relationships between the past (context) and the future (prediction), effectively "copying" relevant visual information from the context frames to guide the generation of the new frames.
*   **Selective Update:** After the DiT predicts the noise $\epsilon_{\phi}(\{\mathbf{z}_t, \mathbf{z}^c\}, \mathbf{p}, t)$, this noise is used to update *only* the noisy latents $\mathbf{z}_t$. The clean context latents $\mathbf{z}^c$ remain unchanged throughout the denoising process.
*   **Positional Encoding:** Since the input sequence length is now variable (prediction length + context length), the model leverages **RoPE (Rotary Position Embedding)**, which can naturally handle variable-length sequences. The original positional encodings are used for the frames being predicted, and new positional encodings are assigned to the concatenated context frames.

### 4.2.3. Memory Retrieval

Using all historical frames as context is computationally infeasible. The `Memory Retrieval` module is designed to select a small, highly relevant subset of frames. This process is visualized in Figure 3 from the paper.

![该图像是示意图，展示了利用历史上下文进行视频生成的框架。左侧展示了上下文学习过程，其中包含无限长度历史上下文以及最新帧的输出。右侧说明了内存检索模块的工作原理，选择高重叠上下文以指导预测帧的生成。](images/3.jpg)
*该图像是示意图，展示了利用历史上下文进行视频生成的框架。左侧展示了上下文学习过程，其中包含无限长度历史上下文以及最新帧的输出。右侧说明了内存检索模块的工作原理，选择高重叠上下文以指导预测帧的生成。*

The paper first dismisses several alternatives:
*   **Random selection:** Ineffective when the history is long.
*   **Neighbor frames (fixed window):** Fails to provide long-term memory and contains redundant information.
*   **Hierarchical compression:** Loses too much information from older frames.

**The proposed method is a camera-trajectory-based search:**
The key idea is to find past frames whose cameras were looking at the same region of space as the camera for the frame being generated.

1.  **Obtaining Camera Trajectories:** Since generation is controlled by user-provided camera poses, the pose for every generated frame is already known. This eliminates the need for a separate camera pose estimation step.

2.  **Determining Co-visibility via FOV Overlap:** The module determines if two frames are "co-visible" by checking for overlap in their Fields of View (FOV). As the experiments constrain camera movement to a 2D plane, this simplifies to a 2D geometry problem.
    *   For two camera poses, the algorithm considers the rays defining the left and right boundaries of their FOVs.
    *   It checks if the rays from the first camera intersect with the rays from the second camera.
    *   A simple rule is used: if both the left-ray pair and the right-ray pair intersect, there is likely an overlap.
    *   To avoid spurious matches (e.g., two cameras facing each other from very far away), it filters out cases where the intersection points are too close or too far from the cameras.
    
        The following figure (Figure 4 from the original paper) illustrates this process.

        ![Fig. 4. Examples of FOV Overlap. We simplify FOV overlap detection to checking intersections between four rays from two camera origins. A practical rule that works for most cases requires: both left and right ray pairs intersect (a, b). However, we must filter out cases where intersection points are either too near (d) or too distant (c) from cameras. While this rule may not cover all scenarios and some corner cases exist (e, f), occasional missed or incorrect candidates don't substantially affect overall performance.](images/4.jpg)
        *该图像是示意图，展示了不同场景下的视域重叠示例。在图中，(a)、(b)和(c)展示了有效的视域重叠，而(d)则显示了视域交点过近或过远的情况。图(e)和(f)展示了一些边界案例，提醒在实际应用中可能会遗漏或错误识别候选帧。*

3.  **Further Filtering:** After the FOV check, if there are still too many candidate frames, additional filtering strategies are applied:
    *   **Non-adjacent selection (`Non-adj`):** From any group of consecutive frames in the filtered set, only one is randomly selected. This reduces redundancy, as adjacent frames are often very similar.
    *   **Spatiotemporal diversity (`Far-space-time`):** The model can be biased to select frames that are more distant from each other in space or time, potentially providing a more comprehensive context.

### 4.2.4. Training and Inference Algorithms

The overall training and inference flows are summarized in the paper's algorithms.

**Algorithm 1: Training Process**
The training process teaches the model how to use context.
$$
ALGORITHM 1: Training Process of Context-as-Memory
Input: Video sequence X and camera annotations C in training dataset, context size k
1: while not converged do
2:   Randomly select predicted video sequence x_pred from X;
3:   Retrieve k-1 frames as context x_c from the rest of X using Memory Retrieval;
4:   Add the first frame of x_pred as the k-th context frame for continuity;
5:   Obtain camera poses {cam_pred, cam_c} for {x_pred, x_c} from C;
6:   Obtain latent embeddings {z_pred, z_c} <- Encoder({x_pred, x_c});
7.   Sample t ~ U(1, T) and ε ~ N(0, I), then corrupt z_pred to z_t;
     8:   Train the model to predict ε from the combined input {z_t, z_c}, prompts, and camera poses using the diffusion loss;
9: end while
$$

**Algorithm 2: Inference Process**
During inference, the model generates video autoregressively, with the history of generated frames serving as the memory bank.
```
ALGORITHM 2: Inference Process of Context-as-Memory
Input: Initial frame set X = {x_init} and camera poses C = {cam_init}
Output: Generated video sequence X
1: while generation not finished do
2:   User provides next target camera pose cam_next;
3:   Retrieve k-1 context frames x_c from X and their poses cam_c by checking FOV overlap with cam_next;
4:   Add the most recent frame from X to the context;
5:   Compute context latent z_c <- Encoder(x_c);
6:   Sample noise z_t ~ N(0, I) and infer the next latent z_next by denoising, conditioned on z_c, cam_c, and cam_next;
7:   Decode generated frames x_next <- Decoder(z_next);
8:   Append x_next to X and cam_next to C;
9: end while
```

## 4.3. Data Collection
A key contribution is a new dataset tailored for this task. Existing datasets with camera poses are typically short video clips. To train for long-term consistency, long videos are needed.
*   **Source:** The dataset was generated using **Unreal Engine 5**.
*   **Content:** It comprises 100 long videos, each 7,601 frames long (at 30 fps, this is over 4 minutes). It features 12 distinct scene styles (cities, countryside, etc.).
*   **Annotations:** Each frame has precise camera extrinsic (pose) and intrinsic parameters. Captions were generated every 77 frames using a multimodal LLM.
*   **Simplification:** To make the problem tractable, camera movement was constrained to a 2D plane (X-Y plane) with rotation only around the vertical Z-axis. This still allows for complex navigation while simplifying the FOV overlap calculation.

# 5. Experimental Setup

## 5.1. Datasets
The experiments exclusively use the custom dataset collected via Unreal Engine 5, as described in the methodology. A 5% split of this dataset, containing scenes not used for training, was held out for testing. The dataset's key features—long duration, diverse scenes, and precise camera annotations—make it ideal for validating the model's memory capabilities.

## 5.2. Evaluation Metrics
The paper uses several metrics to assess video quality and, more importantly, memory capability.

### 5.2.1. Metrics for Memory Capability (Consistency)
*   **PSNR (Peak Signal-to-Noise Ratio)**
    1.  **Conceptual Definition:** PSNR measures the pixel-wise reconstruction quality between two images. It quantifies the ratio between the maximum possible power of a signal and the power of corrupting noise that affects its fidelity. A higher PSNR value indicates a smaller difference between the two images, implying better consistency.
    2.  **Mathematical Formula:**
        \$
        \mathrm{PSNR} = 20 \cdot \log_{10}(\mathrm{MAX}_I) - 10 \cdot \log_{10}(\mathrm{MSE})
        \$
    3.  **Symbol Explanation:**
        *   $\mathrm{MAX}_I$: The maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image).
        *   $\mathrm{MSE}$: The Mean Squared Error between the two images. For two images $I$ and $K$ of size $m \times n$, it is calculated as `\mathrm{MSE} = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j) - K(i,j)]^2`.

*   **LPIPS (Learned Perceptual Image Patch Similarity)**
    1.  **Conceptual Definition:** LPIPS measures the perceptual similarity between two images, which often aligns better with human judgment than pixel-wise metrics like PSNR. It uses features extracted from deep convolutional neural networks (like VGG or AlexNet) that are trained on image classification. If two images are perceptually similar, their feature representations in these networks will be close. A lower LPIPS score indicates greater similarity.
    2.  **Mathematical Formula:**
        \$
        d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} || w_l \odot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l) ||_2^2
        \$
    3.  **Symbol Explanation:**
        *   $d(x, x_0)$: The LPIPS distance between images $x$ and $x_0$.
        *   $l$: Index for the layer in the deep network.
        *   $\hat{y}^l, \hat{y}_0^l$: Feature activations from layer $l$ for images $x$ and $x_0$, normalized channel-wise.
        *   $H_l, W_l$: Height and width of the feature maps at layer $l$.
        *   $w_l$: A set of learned weights to scale the contribution of each channel, training the metric to align better with human perception.
        *   $\odot$: Element-wise multiplication.

### 5.2.2. Metrics for Video Quality
*   **FID (Fréchet Inception Distance)**
    1.  **Conceptual Definition:** FID measures the quality and diversity of a set of generated images compared to a set of real images. It calculates the distance between the feature distributions of the two sets. These features are extracted from a pre-trained InceptionV3 network. A lower FID score indicates that the distribution of generated images is closer to the distribution of real images, implying higher quality and diversity.
    2.  **Mathematical Formula:**
        \$
        \mathrm{FID}(x, g) = ||\mu_x - \mu_g||_2^2 + \mathrm{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})
        \$
    3.  **Symbol Explanation:**
        *   `x, g`: Real and generated data.
        *   $\mu_x, \mu_g$: The mean of the feature vectors for the real and generated images, respectively.
        *   $\Sigma_x, \Sigma_g$: The covariance matrices of the feature vectors for the real and generated images.
        *   $\mathrm{Tr}(\cdot)$: The trace of a matrix (sum of diagonal elements).

*   **FVD (Fréchet Video Distance)**
    1.  **Conceptual Definition:** FVD is the video-domain equivalent of FID. It measures the quality and temporal consistency of generated videos by comparing their feature distributions to those of real videos. The features are extracted from a video classification network (e.g., I3D) pre-trained on a large video dataset. A lower FVD indicates better video quality.
    2.  **Mathematical Formula:** The formula is identical to FID's, but the features are extracted from a video network instead of an image network.

### 5.2.3. Evaluation Approaches
The paper proposes two ways to calculate these metrics to specifically test memory:
1.  **Ground Truth Comparison:** The model is given context frames from a ground truth video and asked to generate a subsequent segment. The generated frames are then compared to the corresponding ground truth frames. This tests if the model can use real context to produce a consistent future.
2.  **History Context Comparison:** This is a more challenging and realistic test. The model generates a long video autoregressively. When the camera trajectory returns to a previously visited location, the newly generated frame is compared to the frame generated *earlier in the same sequence* when the camera was at that location. This directly measures if the model's own generated world is self-consistent.

## 5.3. Baselines
The proposed method, `Context-as-Memory`, is compared against several baselines, which were all implemented on the same base model and trained on the same dataset for a fair comparison:
*   **`1st Frame as Context`:** A minimal baseline using only the very first frame as context.
*   **`1st Frame + Random Context`:** Uses the first frame plus a few randomly selected historical frames.
*   **`DFoT` [Song et al. 2025]:** A strong SOTA baseline that uses a fixed-size window of the most recent frames as context.
*   **`FramePack` [Zhang and Agrawala 2025]:** Another SOTA baseline that hierarchically compresses the entire video history into a few frames.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main comparison results are presented in Table 1 and Figure 5.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th colspan="3">Ground Truth Comparison</th>
<th colspan="3">History Context Comparison</th>
</tr>
<tr>
<th>PSNR↑ LPIPS↓</th>
<th>FID↓</th>
<th>FVD↓</th>
<th>PSNR↑ LPIPS↓</th>
<th>FID↓</th>
<th>FVD↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>1st Frame as Context</td>
<td>15.72 / 0.5282</td>
<td>127.55</td>
<td>937.51</td>
<td>14.53 / 0.5456</td>
<td>157.44</td>
<td>1029.71</td>
</tr>
<tr>
<td>1st Frame + Random Context</td>
<td>17.70 / 0.4847</td>
<td>115.94</td>
<td>853.13</td>
<td>17.07 / 0.3985</td>
<td>119.31</td>
<td>882.36</td>
</tr>
<tr>
<td>DFoT [Song et al. 2025]</td>
<td>17.63 / 0.4528</td>
<td>112.96</td>
<td>897.87</td>
<td>15.70 / 0.5102</td>
<td>121.18</td>
<td>919.75</td>
</tr>
<tr>
<td>FramePack [Zhang and Agrawala 2025]</td>
<td>17.20 / 0.4757</td>
<td>121.87</td>
<td>901.58</td>
<td>15.65 / 0.4947</td>
<td>131.59</td>
<td>974.52</td>
</tr>
<tr>
<td><strong>Context-as-Memory (Ours)</strong></td>
<td><strong>20.22 / 0.3003</strong></td>
<td><strong>107.18</strong></td>
<td><strong>821.37</strong></td>
<td><strong>18.11 / 0.3414</strong></td>
<td><strong>113.22</strong></td>
<td><strong>859.42</strong></td>
</tr>
</tbody>
</table>

**Analysis of Table 1:**
*   **Superior Memory Capability:** `Context-as-Memory` achieves significantly better scores on the memory-focused metrics, PSNR (higher is better) and LPIPS (lower is better), in both evaluation settings. For instance, in the `Ground Truth Comparison`, its PSNR of 20.22 is much higher than the next best (17.70), and its LPIPS of 0.3003 is much lower. This strongly indicates that its `Memory Retrieval` mechanism is effective at finding and utilizing relevant historical frames to maintain scene consistency.
*   **Stronger Performance on `History Context Comparison`:** The performance gap is particularly stark in the more challenging `History Context Comparison` task. `DFoT` and `FramePack`, which rely on recent or compressed context, see a large drop in performance (e.g., DFoT's PSNR drops to 15.70). This is because they cannot recall the appearance of a scene from the distant past. In contrast, `Context-as-Memory` maintains a relatively strong PSNR of 18.11, demonstrating its ability to maintain a self-consistent world.
*   **Improved Video Quality:** The method also achieves the best FID and FVD scores, indicating higher overall video quality. The authors suggest this is because having strong contextual guidance reduces the model's uncertainty and helps prevent the accumulation of errors during long-form generation.

    The qualitative results in Figure 5 visually support these findings, showing that baseline methods fail to reconstruct the original scene after the camera turns back, while the proposed method succeeds.

    ![该图像是图表，展示了与地面真实（GT）比较的结果，标注了不同方法（如C-a-M、Random、DFoT和FramePack）在场景一致性上的表现，包括前向和后向旋转的上下文比较。图中红框标示了不一致的区域，表明当前方法在历史上下文利用上表现优越。](images/5.jpg)
    *该图像是图表，展示了与地面真实（GT）比较的结果，标注了不同方法（如C-a-M、Random、DFoT和FramePack）在场景一致性上的表现，包括前向和后向旋转的上下文比较。图中红框标示了不一致的区域，表明当前方法在历史上下文利用上表现优越。*

## 6.2. Ablation Studies / Parameter Analysis

The authors conducted ablation studies to validate the design choices of their method.

### 6.2.1. Ablation of Context Size
This study investigates how the number of retrieved context frames ($k$) affects performance and speed.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Context Size</th>
<th colspan="2">GT Comp.</th>
<th colspan="2">HC Comp.</th>
<th rowspan="2">Speed (fps)↑</th>
</tr>
<tr>
<th>PSNR↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>LPIPS↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>15.72</td>
<td>0.5282</td>
<td>14.53</td>
<td>0.5456</td>
<td>1.60</td>
</tr>
<tr>
<td>5</td>
<td>17.37</td>
<td>0.4825</td>
<td>15.97</td>
<td>0.5063</td>
<td>1.40</td>
</tr>
<tr>
<td>10</td>
<td>19.14</td>
<td>0.3554</td>
<td>17.75</td>
<td>0.3985</td>
<td>1.20</td>
</tr>
<tr>
<td>20</td>
<td><strong>20.22</strong></td>
<td><strong>0.3003</strong></td>
<td><strong>18.11</strong></td>
<td><strong>0.3414</strong></td>
<td>0.97</td>
</tr>
<tr>
<td>30</td>
<td>20.31</td>
<td>0.3137</td>
<td>18.19</td>
<td>0.3319</td>
<td>0.79</td>
</tr>
</tbody>
</table>

**Analysis of Table 2:**
*   Performance generally improves as the context size increases from 1 to 20, with PSNR and LPIPS scores steadily getting better. This shows that more context provides more useful information for maintaining consistency.
*   However, the generation speed (fps) decreases as the context size grows due to the increased computational load of processing more tokens in the Transformer.
*   The performance gain from 20 to 30 frames is marginal, while the speed drop is significant. This suggests a point of diminishing returns, making a context size of **20** a good trade-off between performance and efficiency.

### 6.2.2. Ablation of Memory Retrieval Strategy
This study analyzes the effectiveness of the different components of the retrieval strategy.

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Strategy</th>
<th colspan="2">GT Comp.</th>
<th colspan="2">HC Comp.</th>
</tr>
<tr>
<th>PSNR↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>LPIPS↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Random</td>
<td>17.70</td>
<td>0.4847</td>
<td>17.07</td>
<td>0.3985</td>
</tr>
<tr>
<td>FOV+Random</td>
<td>19.17</td>
<td>0.3825</td>
<td>17.47</td>
<td>0.3896</td>
</tr>
<tr>
<td>FOV+Non-adj</td>
<td><strong>20.11</strong></td>
<td><strong>0.3075</strong></td>
<td><strong>18.19</strong></td>
<td><strong>0.3571</strong></td>
</tr>
<tr>
<td>FOV+Non-adj+Far-space-time</td>
<td>20.22</td>
<td>0.3003</td>
<td>18.11</td>
<td>0.3414</td>
</tr>
</tbody>
</table>

**Analysis of Table 3:**
*   **`FOV` is critical:** Simply adding the `FOV` overlap filter ($FOV+Random$) provides a massive improvement over the `Random` baseline. This confirms that geometric relevance is far more important than random chance for selecting useful context.
*   **`Non-adj` reduces redundancy:** Adding the `Non-adj` filter (which selects only one frame from a consecutive sequence) provides another significant boost in performance. This shows that filtering out redundant, temporally adjacent frames is an effective strategy.
*   **`Far-space-time` is a minor refinement:** The final addition of the `Far-space-time` heuristic provides only a small additional improvement, suggesting that the first two filters (`FOV` and `Non-adj`) capture most of the necessary logic for effective retrieval.

## 6.3. Open-Domain Results
The paper demonstrates the model's generalization ability by testing it on open-domain images sourced from the internet, which represent styles and scenes not present in the Unreal Engine training data.
The qualitative results in Figure 6 show that even when initialized with these novel images, the model can generate long videos (with a "rotate away and back" trajectory) that maintain scene consistency. This suggests that the model has learned a generalizable skill of utilizing context for memory, not just memorizing the scenes from its training set.

The following figure (Figure 6 from the original paper) shows these open-domain results.

![该图像是插图，展示了三组图像生成示例，每组对应不同的提示内容，如日本风景、黑神话悟空场景和幻想自然景观。这些图像展现了模型在结合历史上下文用于长视频生成中的应用，显示了场景一致性的能力。](images/6.jpg)
*该图像是插图，展示了三组图像生成示例，每组对应不同的提示内容，如日本风景、黑神话悟空场景和幻想自然景观。这些图像展现了模型在结合历史上下文用于长视频生成中的应用，显示了场景一致性的能力。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully identifies and addresses a critical flaw in long video generation: the lack of robust, long-term memory for scene consistency. The proposed `Context-as-Memory` framework offers a simple yet highly effective solution. By treating historical frames directly as memory and introducing an efficient, geometry-aware `Memory Retrieval` module based on FOV overlap, the method achieves state-of-the-art performance in maintaining a coherent world during interactive generation. The work is supported by strong experimental evidence, including a new high-quality dataset, thorough ablations, and demonstrated generalization to open-domain scenarios.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations and outline directions for future research:
*   **Static Scenes:** The current method is designed for and tested on static scenes. Retrieving relevant memory for dynamic scenes (with moving objects, changing lighting, etc.) would be significantly more complex.
*   **Complex Scenarios and Occlusion:** The rule-based FOV overlap check is a heuristic that might fail in complex environments with significant occlusion (e.g., navigating through interconnected rooms in a building), where a direct line of sight does not guarantee co-visibility.
*   **Error Accumulation:** Like all autoregressive generative models, this method is still susceptible to error accumulation over very long generation sequences. While mitigated by strong contextual conditioning, it is not completely solved.
*   **Future Directions:** The authors plan to scale up their approach by applying it to larger base models, which could help address error accumulation and improve open-domain generalization. They also aim to support more complex camera trajectories, larger scenes, and longer generation sequences.

## 7.3. Personal Insights & Critique
*   **Strengths:**
    *   **Simplicity and Elegance:** The core idea of using raw frames as memory and conditioning via simple concatenation is powerful. It avoids the complexities and potential pitfalls of intermediate representations like 3D models or compressed feature spaces.
    *   **Intuitive Retrieval Mechanism:** Using camera geometry (FOV overlap) to find relevant context is a highly intuitive and direct solution to the retrieval problem. It leverages the explicit control signals available in interactive generation.
    *   **Strong Empirical Validation:** The paper is well-supported by a new, relevant dataset, comprehensive experiments, and clear SOTA comparisons. The ablation studies convincingly justify their design choices.

*   **Potential Issues and Areas for Improvement:**
    *   **Scalability of Retrieval:** The current FOV check compares the new camera pose against all previous poses. As the video gets extremely long (e.g., hours of gameplay), this linear search could become a bottleneck. More efficient spatial data structures (like a k-d tree for camera positions) might be needed.
    *   **Dependency on Camera Control:** The method relies entirely on having precise camera poses. It would not be directly applicable to generating long videos from just a text prompt without an explicit camera path, or to tasks like predicting the future of an uncontrolled video.
    *   **Beyond Geometric Overlap:** True scene relevance is more than just geometric overlap. A future system might need to incorporate semantic understanding. For example, if a key object moves, the model should retrieve frames where that object was last seen, even if the camera FOV doesn't overlap perfectly. This could involve combining the geometric search with a content-based search (e.g., using CLIP embeddings).
    *   **Generalization in 3D Space:** The experiments were limited to camera movement on a 2D plane. The effectiveness of the 2D FOV check heuristic in a full 6-DoF (Degrees of Freedom) camera control setting remains an open question and would likely require a more sophisticated 3D geometric analysis.