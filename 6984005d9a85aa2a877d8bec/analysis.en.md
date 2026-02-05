# 1. Bibliographic Information

## 1.1. Title
GameFactory: Creating New Games with Generative Interactive Videos

## 1.2. Authors
-   Jiwen Yu (The University of Hong Kong)
-   Yiran Qin (The University of Hong Kong)
-   Xintao Wang (Kuaishou Technology)
-   Pengfei Wan (Kuaishou Technology)
-   Di Zhang (Kuaishou Technology)
-   Xihui Liu (The University of Hong Kong)

    The authors are a mix of academic researchers from The University of Hong Kong and industry researchers from Kuaishou Technology, a major Chinese tech company known for its short-video platform. This collaboration suggests a strong blend of foundational research and practical application in large-scale generative models.

## 1.3. Journal/Conference
The paper is submitted as a preprint to arXiv. The abstract indicates a publication date of January 14, 2025, and mentions ICLR 2025 for some related works, suggesting it may be under review for a top-tier machine learning conference like ICLR (International Conference on Learning Representations). ICLR is renowned for its focus on deep learning and is considered one of the premier venues in the field.

## 1.4. Publication Year
2025 (as per the preprint metadata).

## 1.5. Abstract
The paper introduces `GameFactory`, a framework designed for generating action-controlled and scene-generalizable game videos. The authors aim to address key challenges in using generative models for game creation. First, they tackle action controllability by creating `GF-Minecraft`, a new action-annotated video dataset from Minecraft that avoids human player bias, and by developing a module for precise keyboard and mouse control. They also extend the model for autoregressive, unlimited-length video generation. The most significant contribution is addressing scene generalization—the ability to create games in diverse styles beyond the training data. To achieve this, `GameFactory` leverages the generative power of pre-trained open-domain video diffusion models. It uses a multi-phase training strategy with a "domain adapter" to decouple the learning of game style (e.g., Minecraft's aesthetic) from the learning of action control. This allows the action control mechanism to be applied to generate interactive videos in any scene the base model can create, representing a major step towards AI-driven game generation.

## 1.6. Original Source Link
-   **Original Source Link:** https://arxiv.org/abs/2501.08325
-   **PDF Link:** https://arxiv.org/pdf/2501.08325v4
-   **Publication Status:** The paper is currently a preprint on arXiv.

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem this paper addresses is the creation of **new, diverse, and interactive games** using generative AI. Modern video diffusion models have shown incredible potential for creating realistic videos, leading to the concept of a "generative game engine"—an AI that can generate playable game worlds on the fly.

However, existing approaches face a critical limitation: **scene generalization**. Current methods are typically trained on data from a single game (e.g., DOOM, Minecraft). When a powerful, pre-trained video model is fine-tuned on this data to learn player controls, it also overfits to the specific visual style of that game. This is often called "style collapse" or "catastrophic forgetting" of open-domain knowledge. The model learns to respond to actions *only* within the Minecraft world, losing its ability to generate videos of a cherry blossom forest or a Renaissance palace. This severely limits the dream of creating entirely new games with unique aesthetics.

The paper's innovative entry point is to treat the learning of **game style** and **action control** as two separate problems. The authors hypothesize that if these two learning processes can be disentangled, the action control knowledge can be learned from a specific game's data, while the model retains its powerful, open-domain scene generation capability from its initial pre-training.

## 2.2. Main Contributions / Findings
The paper makes several key contributions to advance the field of generative game engines:

1.  **A Framework for Scene-Generalizable Game Generation (`GameFactory`):** The primary contribution is a novel framework that can generate action-controllable videos in open-domain scenes, effectively allowing for the creation of new games beyond existing styles.

2.  **A Style-Action Decoupling Strategy:** The core innovation is a multi-phase training strategy that uses a `domain adapter` (implemented with LoRA) to isolate the learning of game-specific style. This allows a separate `action control module` to learn how to respond to player inputs without being tied to that style. During inference, the style adapter is removed, leaving a general-purpose video model that understands game controls.

3.  **An Unbiased Action-Annotated Dataset (`GF-Minecraft`):** To train the action control module effectively, the authors created a 70-hour dataset from Minecraft. Unlike datasets from human gameplay, which are biased towards common actions (e.g., always moving forward), `GF-Minecraft` features a balanced, randomized distribution of atomic actions, enabling the model to respond robustly to rare or unusual command combinations.

4.  **A Versatile Action Control and Generation System:** The framework includes a dedicated module for handling both discrete (keyboard) and continuous (mouse) inputs. It also supports autoregressive generation, allowing it to produce continuous, unlimited-length video streams, which is essential for a playable game experience.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that have become state-of-the-art in generating high-quality images and videos. The core idea is based on a two-step process:
1.  **Forward Process (Noising):** Start with a clean data sample (e.g., an image). Gradually and repeatedly add a small amount of random (Gaussian) noise over a series of timesteps. After many steps, the original image becomes indistinguishable from pure noise. This process is mathematically fixed and does not involve learning.
2.  **Reverse Process (Denoising):** Train a neural network (often a U-Net or a Transformer) to reverse this process. At each timestep, the network takes a noisy image and the current timestep as input and predicts the noise that was added. By subtracting this predicted noise, it can gradually denoise the image, starting from pure random noise and ending with a clean, realistic image.

    **Latent Diffusion Models (LDMs):** To make this process more efficient, especially for high-resolution data like videos, LDMs first compress the video into a smaller, lower-dimensional "latent space" using an autoencoder. The diffusion process then happens in this compressed space, which significantly reduces computational cost. After the denoising is complete, a decoder maps the clean latent representation back to the full-resolution video.

### 3.1.2. Transformers in Vision
The Transformer architecture, originally developed for natural language processing (NLP), has been successfully adapted for vision tasks. Its core component is the **self-attention mechanism**.

-   **Self-Attention:** This mechanism allows the model to weigh the importance of different parts of the input sequence when processing a specific part. For a video, the input is a sequence of image patches or latent frames. Self-attention enables the model to understand complex spatial relationships within a frame (`spatial attention`) and temporal relationships across frames (`temporal attention`), which is crucial for generating coherent motion and action. In this paper, a Transformer-based model serves as the denoising network in the latent diffusion model.

### 3.1.3. LoRA (Low-Rank Adaptation)
`LoRA` is a parameter-efficient fine-tuning (PEFT) technique. Fine-tuning a massive pre-trained model on a new task can be computationally expensive and risks "catastrophic forgetting," where the model loses its original capabilities. `LoRA` addresses this by:
1.  Freezing the weights of the original pre-trained model.
2.  Injecting small, trainable "adapter" modules into the model's layers (typically the attention layers).
3.  These adapters consist of two low-rank matrices whose product approximates the full weight update that would have occurred during normal fine-tuning.

    Since only these small adapter matrices are trained, `LoRA` is much faster and requires less memory. It's particularly effective for learning specific styles or tasks, which can then be "activated" by loading the `LoRA` weights or "deactivated" by not using them, without altering the base model. This paper cleverly uses `LoRA` as a "domain adapter" to capture the visual style of Minecraft.

## 3.2. Previous Works
The paper positions itself against prior work in game video generation:

*   **Early GAN-based models:** Works like `GameGAN` tried to use Generative Adversarial Networks (GANs) for game generation but were limited by the generative quality of GANs at the time.
*   **Game-Specific Diffusion Models:** More recent works have successfully used diffusion models but focused on specific games.
    *   `DIAMOND` and `GameNGen` focused on games like Atari, CS:GO, and DOOM.
    *   `Oasis` and the dataset `VPT` focused on Minecraft.
    *   `PlayGen` worked with Super Mario Bros.
    *   The main drawback of these models is their inability to generalize to new scenes or styles. They are excellent at simulating their source game but cannot create a new one.
*   **Towards Generalization:** The paper acknowledges two very recent works that are moving towards generalization:
    *   `Genie 2`: A large-scale model that shows impressive results, but its success relies on collecting massive amounts of action-labeled data, which is expensive and difficult to scale.
    *   `Matrix`: Shows promise in generalizing controls for racing games, but the action space (turn left/right, accelerate) is much simpler than the first-person navigation (7 keys + mouse) tackled by `GameFactory`.

## 3.3. Technological Evolution
The field of generative games has evolved from simple procedural content generation to sophisticated AI-driven world simulation.
1.  **Early GANs:** Limited to simpler, lower-resolution game environments.
2.  **Rise of Diffusion Models:** Enabled high-fidelity video generation, leading to models that could simulate existing games with impressive visual quality.
3.  **Current Frontier (Scene Generalization):** The focus is now shifting from merely *simulating* existing games to *creating* entirely new ones. This requires leveraging the vast, open-domain knowledge of large pre-trained models and finding ways to teach them interactive controls without sacrificing their creative flexibility. `GameFactory` is a prime example of this new direction.

## 3.4. Differentiation Analysis
`GameFactory`'s core innovation lies in its **decoupling strategy**, which sets it apart from previous methods:

*   **Standard Fine-tuning (e.g., `Oasis`, `GameNGen`):** These methods take a pre-trained model and fine-tune it directly on action-annotated game data. This entangles the learning of style and action control, leading to style collapse. The model learns that "moving forward" looks like "moving forward *in Minecraft*."
*   **GameFactory's Decoupled Approach:** `GameFactory` uses separate components to learn style and actions.
    *   The **`LoRA` adapter** is responsible for learning the Minecraft style.
    *   The **`action control module`** is responsible for learning the physics of movement (e.g., pressing 'W' moves the camera forward).
        Because these are trained in separate phases with frozen components, the action control module learns a more abstract, style-independent representation of control. This allows it to be combined with the base open-domain model (without the `LoRA` adapter) to control movement in any scene, not just Minecraft.

---

# 4. Methodology

## 4.1. Principles
The core principle of `GameFactory` is to achieve **scene-generalizable action control** by **disentangling style learning from action control learning**. The intuition is that the "physics" of first-person movement (e.g., what happens when you press 'W' or move the mouse) is a general concept that can be learned from one environment (like Minecraft) and applied to others. However, if the model learns this concept while also learning the specific visual style of Minecraft, the two become inseparable. By using a modular approach with a dedicated `domain adapter` for style and an `action control module` for actions, the framework can learn both and then discard the style-specific part during inference to achieve generalization.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Preliminaries: Backbone Model
The framework is built on a transformer-based latent video diffusion model. A video sequence $\mathbf{X}$ is first compressed by an encoder $E(\cdot)$ into a latent representation $\mathbf{Z} = E(\mathbf{X})$. This compression is both spatial (per frame) and temporal. A video of $(1 + rn)$ frames is compressed into $(1+n)$ latent frames, where $r$ is the temporal compression ratio. The training objective is to teach the model to predict the noise $\epsilon$ added to a clean latent $\mathbf{Z}_0$ to create a noisy latent $\mathbf{Z}_t$ at timestep $t$. When conditioned on a text prompt $\mathbf{p}$ and a sequence of actions $\mathbf{A}$, the loss function is:

$$
\mathcal{L}_{\mathbf{a}}(\phi) = \mathbb{E}[||\epsilon_{\phi}(\mathbf{Z}_t, \mathbf{p}, \mathbf{A}, t) - \epsilon||_2^2]
$$

-   $\phi$: The parameters of the denoising model.
-   $\mathbf{Z}_t$: The noisy latent video at timestep $t$.
-   $\mathbf{p}$: The text prompt condition.
-   $\mathbf{A}$: The sequence of player actions.
-   $t$: The current noise timestep.
-   $\epsilon$: The ground truth random noise that was added.
-   $\epsilon_{\phi}(\cdot)$: The neural network (the Transformer) that predicts the noise.

    During inference, the model starts with pure noise $\mathbf{Z}_T$ and iteratively denoises it using the learned network to produce a clean latent $\mathbf{Z}_0$, which is then decoded back into a video $\mathbf{X}$ by the decoder $D(\cdot)$.

### 4.2.2. Action-Controlled Video Generation

This section details the components needed to make the video generation controllable by player actions.

#### **GF-Minecraft Dataset**
A key component is the training data. The authors created the `GF-Minecraft` dataset with three crucial properties:
1.  **Accessible Data Source:** Minecraft provides an API (`MineDojo`) that allows for programmatic control and data collection, making it easy to gather large amounts of action-annotated video.
2.  **Unbiased Actions:** Unlike datasets from human players (`VPT`), which are heavily biased (e.g., players mostly move forward), this dataset was generated by executing randomized sequences of atomic actions (e.g., press 'W' for a random duration, then 'space', then move mouse). This ensures the model sees a balanced distribution of all possible actions and can respond to rare commands.
3.  **Diverse Scenes with Text:** Videos were captured in various scenes, weather, and times of day. A multimodal LLM (`MiniCPM`) was used to automatically generate textual descriptions for each video clip, providing rich conditioning for the model.

#### **Action Control Module**
This module is injected into the Transformer blocks of the denoising network to fuse action information with the video features. The architecture is shown in Figure 3 of the paper.

![Figure 3. (a) Integration of Action Control Module into transformer blocks of the video diffusion model. (b) Different control mechanisms for continuous mouse and discrete keyboard inputs.](images/6.jpg)

**Step 1: Grouping Actions with a Sliding Window**
A challenge arises from the temporal compression of the video. If the compression ratio $r=4$, there are `4` times as many action inputs as there are latent video frames. To align them, actions are grouped. For the $i$-th latent frame, a window of recent actions is considered, specifically from action $r \times (i - w + 1)$ to `ri`, where $w$ is the window size. This also helps capture delayed effects (e.g., pressing 'jump' affects the video for several subsequent frames). This process is visualized in Figure 4.

![Figure 4. Due to temporal compression (compression ratio $r = 4$ ), the number of latent features differs from the number of actions, causing granularity mismatch during fusion. Grouping aligns these sequences for fusion. Additionally, the $i$ -th latent feature can fuse with action groups within a previous window (window size $w = 3$ ), accounting for delayed action effects (e.g., 'jump' key affects several subsequent frames).](images/7.jpg)

**Step 2: Fusing Action Signals**
The module handles continuous mouse movements and discrete keyboard inputs differently, as shown in Figure 3(b).
*   **Mouse Movements Control (Continuous):** The grouped mouse actions $\mathbf{M}_{group}$ (a sequence of continuous values) are processed via **concatenation**. They are reshaped, repeated to match the spatial dimensions of the video features $\mathbf{F}$, and then concatenated along the channel dimension. This combined feature tensor is then passed through an MLP and a temporal self-attention layer to integrate the continuous control signals.
*   **Keyboard Actions Control (Discrete):** The grouped keyboard actions $\mathbf{K}_{group}$ (a sequence of discrete keys) are first converted to embeddings. These action embeddings are then fused using **cross-attention**. The video features $\mathbf{F}$ act as the `query`, while the keyboard action embeddings serve as the `key` and `value`. This is analogous to how text prompts are integrated into diffusion models and is effective for categorical inputs.

#### **Autoregressive Generation for Long Videos**
To create a continuous, playable game, the model must generate videos of unlimited length. The authors achieve this with an autoregressive approach inspired by `Diffusion Forcing`.

The process is illustrated in Figure 5 of the paper.

![Figure 5. Illustration of autoregressive video generation. The frames from index 0 to $k$ serve as conditional frames, while the remaining $N - k$ frames are for prediction, with $k$ randomly selected. (a) Training stage: Loss computation and optimization focus only on the noise of predicted frames. (b) Inference stage: The model iteratively selects the latest $k + 1$ frames as conditions to generate $N - k$ new frames, enabling autoregressive generation.](images/8.jpg)
*该图像是示意图，展示了自回归视频生成的过程。在（a）训练阶段，图示说明了噪声和真实潜在视频帧的处理；而在（b）推理阶段，模型通过历史视频潜在信息进行自回归生成。此过程允许根据前 $k + 1$ 帧条件生成剩余的 $N - k$ 帧。图中还提及了视频扩散模型和训练损失的计算。*

*   **Training:** During training, a video sequence of $N+1$ latent frames is used. A random integer $k$ is chosen. The first $k+1$ frames are treated as "context" or "history" and are kept clean (no noise). Noise is added only to the remaining `N-k` frames. The model's task is to predict the noise for these future frames, conditioned on the clean past frames. The loss is calculated **only on the predicted frames** (`N-k` frames), which was found to be more effective than calculating loss on all frames.
*   **Inference:**
    1.  The model first generates an initial sequence of $N+1$ frames.
    2.  To generate the next part of the video, it takes the last $k+1$ frames from the generated sequence as the new context.
    3.  It then denoises a new set of random latents to produce the next `N-k` frames.
    4.  These new frames are appended to the video, and the process repeats, enabling the generation of an arbitrarily long video stream. This is more efficient than single-frame prediction as it generates multiple frames per step.

### 4.2.3. Open-Domain Game Scene Generalization

This is the core contribution for creating *new* games. The process involves a carefully designed multi-phase training strategy to decouple style from action control.

![该图像是一个示意图，展示了GameFactory框架的多阶段训练过程，包含四个阶段：开放域数据、Minecraft游戏数据、行动控制以及开放域结果。每个阶段分别阐述了数据输入、模型训练与产生的结果，呈现出通过解耦实现场景通用的行动控制的流程。](images/9.jpg)
*该图像是一个示意图，展示了GameFactory框架的多阶段训练过程，包含四个阶段：开放域数据、Minecraft游戏数据、行动控制以及开放域结果。每个阶段分别阐述了数据输入、模型训练与产生的结果，呈现出通过解耦实现场景通用的行动控制的流程。*

#### **Phase #0: Pre-trained Model**
The starting point is a large, pre-trained text-to-video diffusion model with strong open-domain generative capabilities. This model knows how to generate a wide variety of scenes but has no concept of interactive control.

#### **Phase #1: Tune LoRA to Fit Game Videos (Style Learning)**
In this phase, the goal is to capture the visual style of Minecraft.
*   The base pre-trained model's weights are **frozen**.
*   A `LoRA` adapter is added to the model.
*   Only the `LoRA` weights are trained on the `GF-Minecraft` video dataset (without action labels).
    This forces the small `LoRA` adapter to learn the specific stylistic features of Minecraft (e.g., blocky textures, specific color palette).

#### **Phase #2: Tune Action Control Module (Action Learning)**
Here, the goal is to learn the response to player actions.
*   The base model's weights and the `LoRA` weights from Phase #1 are both **frozen**.
*   The newly introduced `action control module` is the only component that is trained.
*   The model is trained on the `GF-Minecraft` dataset, this time using both the videos and their corresponding action labels.
    Since the base model provides general video structure and the `LoRA` adapter provides the Minecraft style, the diffusion loss can only be minimized by the `action control module` learning the correct mapping between actions and video changes. This isolates the learning of action dynamics.

#### **Phase #3: Inference on Open Domain (Generalization)**
This is the final step where the magic happens.
*   The base pre-trained model from Phase #0 is used.
*   The trained `action control module` from Phase #2 is plugged in.
*   Crucially, the `LoRA` adapter from Phase #1 is **removed**.

    The result is a model that has the open-domain generative power of the original model but is now controllable via the action module. It can respond to keyboard and mouse inputs to navigate through any scene it can generate (e.g., a photorealistic forest), not just Minecraft.

---

# 5. Experimental Setup

## 5.1. Datasets
*   **`GF-Minecraft`:** The primary dataset created by the authors. It consists of **70 hours** of gameplay video collected from Minecraft using the `MineDojo` framework. Its key feature is the unbiased action distribution achieved by randomizing atomic actions. 5% of the data was held out for testing. The test set was further divided into three subsets for ablation studies:
    *   `only-key`: Videos with only keyboard actions.
    *   `mouse-small`: Videos with small mouse movements.
    *   `mouse-large`: Videos with large, sweeping mouse movements.
*   **`VPT`:** A publicly available Minecraft dataset collected from human gameplay. It was used as a point of comparison to demonstrate the negative effect of human action bias on model training. The authors used the "Find Cave" subset as it is most similar to their navigation-focused task.

## 5.2. Evaluation Metrics

The paper uses a comprehensive set of metrics to evaluate video quality, semantic consistency, and action-following fidelity.

1.  **Flow:**
    *   **Conceptual Definition:** This metric measures the model's ability to follow an action by comparing the motion in the generated video to the motion in the ground truth video. It calculates the optical flow (a representation of pixel movement between frames) for both videos and then computes the mean squared error (MSE) between them. A lower `Flow` score means the generated motion is more similar to the reference motion.
    *   **Mathematical Formula:** $ \text{Flow} = \frac{1}{H \times W \times C} \sum_{i,j,k} (O_{gen}(i,j,k) - O_{ref}(i,j,k))^2 $
    *   **Symbol Explanation:**
        *   $O_{gen}$: Optical flow of the generated video.
        *   $O_{ref}$: Optical flow of the reference video.
        *   `H, W, C`: Height, width, and channels of the optical flow field.

2.  **Cam:**
    *   **Conceptual Definition:** This metric evaluates how accurately the generated video reflects the camera movement implied by the control inputs. It uses an external tool (`GLOMAP`) to estimate the 3D camera pose (position and orientation) for each frame in both the generated and reference videos. It then calculates the Euclidean distance between these camera pose trajectories. A lower `Cam` score indicates better action-following.
    *   **Mathematical Formula:** $ \text{Cam} = \sqrt{\sum_{t=1}^{N} ||P_{gen}(t) - P_{ref}(t)||_2^2} $
    *   **Symbol Explanation:**
        *   $P_{gen}(t)$: Camera pose at frame $t$ in the generated video.
        *   $P_{ref}(t)$: Camera pose at frame $t$ in the reference video.
        *   $N$: Total number of frames.

3.  **CLIP Score (↑):**
    *   **Conceptual Definition:** This metric assesses the semantic relevance of the generated video to the input text prompt. It uses the pre-trained CLIP model to compute embeddings for the text prompt and for each frame of the generated video. The cosine similarity between the text and frame embeddings is then calculated and averaged. A higher `CLIP` score indicates better alignment with the text prompt.
    *   **Mathematical Formula:** `\text{CLIP Score} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_T(p), E_I(x_i))`
    *   **Symbol Explanation:**
        *   $E_T$: CLIP text encoder.
        *   $E_I$: CLIP image encoder.
        *   $p$: The input text prompt.
        *   $x_i$: The $i$-th frame of the generated video.
        *   $N$: Total number of frames.

4.  **FID (Fréchet Inception Distance) (↓):**
    *   **Conceptual Definition:** `FID` measures the quality and diversity of generated images (or video frames) by comparing their feature distribution to that of real images. It uses a pre-trained InceptionV3 network to extract features. A lower `FID` score means the distribution of generated frames is closer to the distribution of real frames, indicating higher quality and diversity.
    *   **Mathematical Formula:** $ \text{FID}(x, g) = ||\mu_x - \mu_g||^2_2 + \text{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2}) $
    *   **Symbol Explanation:**
        *   $\mu_x, \mu_g$: Mean of the feature vectors for real and generated data, respectively.
        *   $\Sigma_x, \Sigma_g$: Covariance matrices of the feature vectors.
        *   $\text{Tr}(\cdot)$: The trace of a matrix.

5.  **FVD (Fréchet Video Distance) (↓):**
    *   **Conceptual Definition:** `FVD` is the video-domain equivalent of `FID`. It measures the quality and temporal coherence of generated videos by comparing their feature distributions to real videos. The features are extracted using a pre-trained video classification model (e.g., I3D). A lower `FVD` indicates better video quality.

6.  **Dom (↑):**
    *   **Conceptual Definition:** This is a custom metric introduced by the authors to measure domain preservation. It calculates the CLIP space similarity between videos generated by the original pre-trained model and the fine-tuned `GameFactory` model (for the same text prompt). A higher `Dom` score indicates that the fine-tuned model has not drifted far from the original model's domain, meaning less style leakage has occurred.

## 5.3. Baselines
The paper's main comparisons are not against other full frameworks but are designed to validate its own components and strategies:
*   **Ablation Study Baselines:** Different designs for the `action control module` (e.g., `cross-attention` vs. `concatenation` for different input types) are compared against each other.
*   **Training Strategy Baseline:** The proposed **`multi-phase`** training strategy is compared against a **`one-phase`** strategy where the `LoRA` adapter and `action control module` are trained simultaneously. This baseline is crucial for proving the effectiveness of the decoupling approach.
*   **Dataset Baseline:** The model trained on their `GF-Minecraft` dataset is compared against a model trained on the human-biased `VPT` dataset.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results strongly support the paper's claims. The multi-phase decoupling strategy is shown to be superior for achieving scene generalization, and the `GF-Minecraft` dataset proves more effective for training robust action control than existing biased datasets.

### 6.1.1. Action Controllability Ablation
The authors first tested different designs for the action control module. The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th colspan="2">Control Module</th>
<th colspan="5">Only-Key</th>
<th colspan="5">Mouse-Small</th>
<th colspan="5">Mouse-Large</th>
</tr>
<tr>
<th>Key</th>
<th>Mouse</th>
<th>Cam↓</th>
<th>Flow↓</th>
<th>CLIP↑</th>
<th>FID↓</th>
<th>FVD↓</th>
<th>Cam↓</th>
<th>Flow↓</th>
<th>CLIP↑</th>
<th>FID↓</th>
<th>FVD↓</th>
<th>Cam↓</th>
<th>Flow↓</th>
<th>CLIP↑</th>
<th>FID↓</th>
<th>FVD↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Cross-Attn</td>
<td>Cross-Attn</td>
<td>0.0527</td>
<td>8.67</td>
<td>0.3313</td>
<td>107.13</td>
<td>814.05</td>
<td>0.0798</td>
<td>20.46</td>
<td>0.3137</td>
<td>125.67</td>
<td>1203.29</td>
<td>0.1362</td>
<td>325.18</td>
<td>0.3103</td>
<td>167.37</td>
<td>1383.92</td>
</tr>
<tr>
<td>Concat</td>
<td>Concat</td>
<td>0.0853</td>
<td>22.37</td>
<td>0.3277</td>
<td>103.89</td>
<td>786.50</td>
<td>0.0756</td>
<td>19.18</td>
<td>0.3159</td>
<td>133.42</td>
<td>1151.71</td>
<td>0.1179</td>
<td>258.93</td>
<td>0.3123</td>
<td>145.74</td>
<td>1405.47</td>
</tr>
<tr>
<td>Cross-Attn</td>
<td>Concat</td>
<td><strong>0.0439</strong></td>
<td><strong>7.79</strong></td>
<td>0.3292</td>
<td>105.28</td>
<td>795.03</td>
<td><strong>0.0685</strong></td>
<td><strong>18.64</strong></td>
<td><strong>0.3184</strong></td>
<td>127.84</td>
<td><strong>1032.98</strong></td>
<td><strong>0.1021</strong></td>
<td><strong>249.54</strong></td>
<td>0.3107</td>
<td>139.91</td>
<td>1420.89</td>
</tr>
</tbody>
</table>

**Analysis:**
*   The best-performing combination (bottom row) is using **`Cross-Attention` for discrete keyboard inputs** and **`Concatenation` for continuous mouse inputs**.
*   For keyboard (`Only-Key` test set), `Cross-Attn` is clearly superior to `Concat` (Cam score 0.0439 vs. 0.0853). The authors suggest this is because cross-attention's similarity-based mechanism is well-suited for categorical signals, similar to text.
*   For mouse movements, `Concat` outperforms `Cross-Attn`. This is likely because concatenation directly preserves the magnitude of the continuous movement values, whereas cross-attention might diminish this information through its similarity calculations.
*   The qualitative results in Figure 7 further support this, showing that using concatenation for key inputs leads to poor action following.

    ![该图像是示意图，展示了不同输入控制方法下生成的视频效果。图中包括四个区块，分别对比了使用“Mouse”和“Key”控制的不同组合（如“Concat”和“Cross-Attn”）所产生的场景。这些场景显示了在动作控制中产生的多种结果，其中标记为“BAD Results”的区块突出了较差的效果，为研究者提供了可视化的比较依据。](images/10.jpg)
    *该图像是示意图，展示了不同输入控制方法下生成的视频效果。图中包括四个区块，分别对比了使用“Mouse”和“Key”控制的不同组合（如“Concat”和“Cross-Attn”）所产生的场景。这些场景显示了在动作控制中产生的多种结果，其中标记为“BAD Results”的区块突出了较差的效果，为研究者提供了可视化的比较依据。*

### 6.1.2. Scene Generalization Results
This is the most critical experiment. The authors compare their `multi-phase` training strategy with a `one-phase` baseline. The results are shown in Table 3.

The following are the results from Table 3 of the original paper:

| Strategy | Domain | Cam↓ | Flow↓ | Dom↑ | CLIP↑ | FID↓ | FVD↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Multi-Phase | In- | 0.0839 | 43.48 | - | - | - | - |
| Multi-Phase | Open- | **0.0997** | **54.13** | **0.7565** | **0.3181** | **121.18** | **1256.94** |
| One-Phase | Open- | 0.1134 | 76.02 | 0.7345 | 0.3111 | 167.79 | 1323.58 |

**Analysis:**
*   The `multi-phase` strategy significantly outperforms the `one-phase` approach in the open-domain setting across all metrics.
*   **Action Following (`Cam`↓, `Flow`↓):** The `multi-phase` model follows actions more accurately, with scores closer to the in-domain baseline.
*   **Domain Preservation (`Dom`↑):** The higher `Dom` score for the `multi-phase` model confirms that it stays closer to the original model's generative domain, meaning less style leakage from Minecraft.
*   **Generation Quality (`CLIP`↑, `FID`↓, `FVD`↓):** The `multi-phase` model also produces higher quality videos that are better aligned with the text prompts.
*   The qualitative comparison in Figure 8 shows this visually. The `one-phase` model's output is heavily biased towards a Minecraft-like style (e.g., blocky trees), while the `multi-phase` model preserves the photorealistic style of the original model.

    ![该图像是一个示意图，展示了原始模型和经过多相位训练及单相位训练的开放领域结果。图中包括樱花森林场景，展示不同训练方法在动作控制和场景生成上的表现。](images/11.jpg)
    *该图像是一个示意图，展示了原始模型和经过多相位训练及单相位训练的开放领域结果。图中包括樱花森林场景，展示不同训练方法在动作控制和场景生成上的表现。*

The various game videos generated in open-domain scenes, as shown in Figure 1, demonstrate the success of this approach.

![该图像是示意图，展示了GameFactory框架中的互动视频生成过程。上方展示了在不同场景下，玩家通过按键操作（W、A、S、D及空格）控制角色的动态画面，底部则描述了相应的操作指令，以实现有效的场景控制和动作生成。](images/1.jpg)
*该图像是示意图，展示了GameFactory框架中的互动视频生成过程。上方展示了在不同场景下，玩家通过按键操作（W、A、S、D及空格）控制角色的动态画面，底部则描述了相应的操作指令，以实现有效的场景控制和动作生成。*

### 6.1.3. GF-Minecraft Dataset Evaluation
To validate their dataset design, the authors compare a model trained on `GF-Minecraft` with one trained on the human-biased `VPT` dataset.

The following are the results from Table 4 of the original paper:

| Dataset | Cam↓ | Flow↓ | CLIP↑ | FID↓ | FVD↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| VPT [3] | 0.1324 | 107.67 | 0.3174 | 156.69 | 1233.15 |
| GF-Minecraft (ours) | **0.0839** | **43.48** | 0.3135 | **125.85** | **1047.59** |

Table 5 shows the stark difference in action distribution:

The following are the results from Table 5 of the original paper:

| Dataset | W | A | S | D | Space | Shift | Ctrl |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VPT [3] | 50.11% | 4.03% | 0.32% | 3.45% | 20.37% | 0.14% | 19.58% |
| Ours | 13.56% | 13.56% | 13.56% | 13.56% | 15.25% | 15.25% | 15.25% |

**Analysis:**
*   The model trained on `GF-Minecraft` shows vastly superior action-following performance (`Cam` and `Flow` scores are much lower).
*   The reason is evident in Table 5: the `VPT` dataset is extremely biased. The 'S' key (move backward) is used only 0.32% of the time. A model trained on this data will be very poor at executing backward movements. `GF-Minecraft` has a much more balanced distribution.
*   Figure 9 provides a powerful qualitative example. When asked to perform rare actions like "jump in place" or "move backward," the `VPT`-trained model fails, often adding a forward motion. The `GF-Minecraft`-trained model executes these commands correctly, demonstrating its robustness.

    ![Figure 9. Compare the dataset on actions that are less commonly used by human players to test the effect of human bias in dataset.](images/12.jpg)
    *该图像是图表，展示了在不同动作控制方法（VPT与本文方法）的对比效果。图中包含两个动作示例：“按住S键向后移动”和“按空格键原地跳跃”。每组展示了不同方法在执行这些动作时生成的场景效果，突出展示了本文方法在动作控制的优势和表现。*

### 6.1.4. Long Video Generation Evaluation
The authors tested the importance of their loss calculation strategy for autoregressive generation.

The following are the results from Table 6 of the original paper:

| Loss Scope | Cam↓ | Flow↓ | CLIP↑ | FID↓ | FVD↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| All frames | 0.1547 | 148.73 | 0.2965 | 176.07 | 1592.43 |
| Only predicted frames | **0.0924** | **85.45** | **0.3190** | **136.95** | **1154.45** |

**Analysis:**
*   Calculating the loss **only on the frames that need to be predicted** yields significantly better results across all metrics.
*   The authors reason that this prevents the model from being distracted by trying to learn from the "noise" of previously generated frames, which is irrelevant to the task of predicting the future.
*   Figure 10 demonstrates that the model can successfully generate long, coherent video sequences of over 300 frames.

    ![Figure 10. Demonstration of key frames in generated long video.](images/13.jpg)
    *该图像是示意图，展示了生成的长视频中的关键帧。上半部分显示从第1帧到第151帧的关键画面，底部则展示第176帧到第326帧的画面，体现了动作控制的演变和场景变化。*

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper presents `GameFactory`, a significant step towards creating generative game engines that can produce new, interactive experiences in diverse, open-domain settings. The authors successfully address the critical challenge of scene generalization by introducing a novel **style-action decoupling strategy**. This method, which combines a `LoRA`-based domain adapter with a multi-phase training schedule, allows an action control module to learn from game-specific data without causing the base model to overfit to that game's visual style. Furthermore, the paper contributes the `GF-Minecraft` dataset, whose unbiased action distribution proves crucial for training robustly controllable models, and demonstrates an effective method for generating unlimited-length videos.

## 7.2. Limitations & Future Work
The authors acknowledge that this is an early step and that many challenges remain for creating a fully-fledged generative game engine. They highlight several areas for future work:
*   **Complex Gameplay and Objectives:** `GameFactory` focuses on navigation and interaction with the environment's physics. It does not yet address higher-level game mechanics like quest design, non-player character (NPC) interaction, or object manipulation (e.g., picking up items, crafting).
*   **Player Feedback Systems:** A true game needs to respond to player success and failure. Integrating feedback loops and dynamic difficulty adjustment is a major future challenge.
*   **Long-Context Memory:** While the model can generate long videos, maintaining long-term consistency (e.g., remembering that a door was opened 10 minutes ago) remains an open problem for generative models.
*   **Real-Time Generation:** Diffusion models are notoriously slow to sample from. Achieving real-time generation speeds necessary for interactive gameplay is a significant engineering and research hurdle.

## 7.3. Personal Insights & Critique
`GameFactory` offers a powerful and elegant solution to the problem of style collapse in conditional generative models.

**Strengths and Inspirations:**
*   The **decoupling principle** is the paper's most brilliant insight. It is a highly generalizable idea that could be applied to many other domains. For instance, one could train a model to understand human poses from a dataset of dancers in a specific style (e.g., ballet) and then use that pose control on an open-domain model to animate a character doing ballet in space, without the output looking like a ballet studio.
*   The creation of the **unbiased dataset** is a methodologically sound and critical contribution. It highlights a common pitfall in machine learning: biased training data leads to biased models. Their solution of generating data from randomized atomic actions is a valuable lesson for anyone building controllable systems.
*   The work moves the conversation from "can we simulate this game?" to "can we create any game?", which is a far more exciting and ambitious goal.

**Potential Issues and Areas for Improvement:**
*   **Complexity of Control:** While the action space is more complex than some prior work, it still represents a simplified version of modern game controls. Integrating more complex interactions, such as inventory management or aiming with a crosshair, would be a necessary next step.
*   **World Model Coherence:** The generated world is still a "local" simulation. Each new segment of video is generated based on a short history. This can lead to a lack of global coherence; for example, walking in a circle might not bring you back to the exact same starting point. A true world model would need a more persistent internal representation of the environment.
*   **Scalability of Decoupling:** The decoupling approach works well for a single game style (Minecraft). It remains to be seen how well it would scale if one wanted to learn action controls from multiple different games, each with its own physics and style. Would a single action module suffice, or would one need a more complex, context-aware control system?

    Overall, `GameFactory` is a foundational piece of research that lays out a clear and effective path for overcoming one of the biggest obstacles in AI-driven game generation. It successfully combines existing powerful tools (`Transformers`, `diffusion models`, `LoRA`) in a novel way to solve a well-defined and important problem.