# 1. Bibliographic Information

## 1.1. Title
FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios

The title clearly states the paper's primary goal: to achieve flexible control over actions in video generation, particularly in "heterogeneous scenarios." This implies situations where the subject performing the action and the context may differ significantly between the source of the action and the target of the generation.

## 1.2. Authors
- Shiyi Zhang (Tsinghua Shenzhen International Graduate School, Tsinghua University)
- Junhao Zhuang (Tsinghua Shenzhen International Graduate School, Tsinghua University)
- Zhaoyang Zhang (Tencent ARC Lab)
- Ying Shan (Tencent ARC Lab)
- Yansong Tang (Tsinghua Shenzhen International Graduate School, Tsinghua University)

  The authors are from prestigious academic institutions (Tsinghua University) and a leading industrial research lab (Tencent AI Lab - ARC), indicating a strong combination of academic rigor and industry-relevant application focus.

## 1.3. Journal/Conference
Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers '25)

SIGGRAPH is the premier international conference on computer graphics and interactive techniques. It is widely considered the most prestigious venue in the field, known for showcasing groundbreaking research. Publication at SIGGRAPH signifies a high level of innovation, technical quality, and impact.

## 1.4. Publication Year
2025

## 1.5. Abstract
The abstract introduces the task of **action customization**: generating a video where a subject performs an action specified by a control signal. The authors identify a key limitation in current methods: they impose strict constraints on spatial consistency (e.g., layout, skeleton, viewpoint) between the source action and the target subject, which limits their use in diverse, real-world scenarios.

To solve this, the paper proposes **FlexiAct**, a method to transfer an action from a reference video to an arbitrary target image, even when their layouts, viewpoints, and skeletal structures differ. The core challenges are precise action control, spatial adaptation, and maintaining the target subject's identity.

FlexiAct introduces two novel components:
1.  **RefAdapter**: A lightweight adapter that helps the model adapt the action to the target's spatial structure while preserving its appearance.
2.  **Frequency-aware Action Extraction (FAE)**: A new technique that extracts motion by leveraging an observation that the video generation process (denoising) naturally focuses on motion (low-frequency information) in its early stages and on appearance details (high-frequency information) in its later stages.

    Experiments confirm that FlexiAct successfully transfers actions across subjects with diverse characteristics. The authors also release their code and models to facilitate further research.

## 1.6. Original Source Link
- **Original Source Link:** `https://arxiv.org/abs/2505.03730`
- **PDF Link:** `https://arxiv.org/pdf/2505.03730v1.pdf`
- **Publication Status:** This paper is a preprint submitted to SIGGRAPH 2025. As of the current date, its final publication status is pending peer review and conference acceptance.

# 2. Executive Summary

## 2.1. Background & Motivation
**Core Problem:** The paper addresses the problem of **action transfer** in video generation. The goal is to make a subject from a static image perform an action seen in a reference video. The central challenge lies in what the authors term "heterogeneous scenarios"—situations where the subject in the target image and the subject in the reference video have different body shapes, skeletal structures, viewpoints, or are positioned differently in the frame (layout).

**Importance and Gaps:** Existing methods for controlling motion in video generation fall short in these scenarios:
*   **Predefined Signal Methods** (e.g., pose-guided): These methods, like `AnimateAnyone`, rely on extracting a skeleton (pose) from the reference video and applying it to the target image. This approach fails when:
    *   The target subject has a different skeleton from the source (e.g., human to animal, or even two humans with very different body types).
    *   The viewpoint is drastically different, making pose alignment impossible.
    *   The subject is non-humanoid (e.g., an animal, a cartoon character), for which reliable pose estimators do not exist.
*   **Global Motion Methods** (e.g., `MotionDirector`): These methods capture the overall motion dynamics of a video (like camera movement or general object flow) but struggle to apply that motion specifically to a new subject while preserving its identity and adapting to its unique structure. They tend to generate videos that mimic the layout of the reference video rather than animating a specific character from a target image.

    This gap makes most existing tools inflexible and unsuitable for creative applications where a user might want to, for instance, make a drawing of a dog perform a dance from a video of a person.

**Entry Point/Innovative Idea:** The paper's innovative idea is to decouple the **action** from the **spatial structure** and **appearance**. They propose a framework, `FlexiAct`, that can "understand" the motion in a reference video and intelligently "reinterpret" it for a new character, all while ensuring the new character looks consistent. This is achieved through two key technical innovations: `RefAdapter` for spatial adaptation and `FAE` for a novel, in-process method of action extraction.

## 2.2. Main Contributions / Findings
The paper presents the following main contributions:

1.  **FlexiAct Framework:** A novel Image-to-Video (I2V) framework designed specifically for flexible action transfer in heterogeneous scenarios. It is the first, according to the authors, to successfully adapt actions to subjects with diverse spatial structures while maintaining high fidelity to both the action and the target's appearance.

2.  **RefAdapter (Reference Adapter):** A lightweight and efficient module that enables the video generation model to adapt to different spatial structures. It conditions the generation on a target image, preserving the subject's appearance, but does so flexibly, avoiding the rigid constraints of previous methods. It achieves fine-grained control comparable to computationally expensive methods like `ReferenceNet` but with far fewer trainable parameters.

3.  **FAE (Frequency-aware Action Extraction):** A groundbreaking method for extracting action information. Instead of using a separate network to process the video for motion, FAE works *during* the video generation (denoising) process. It is based on the insight that early denoising steps naturally focus on low-frequency signals (coarse motion), while later steps focus on high-frequency signals (fine details). FAE dynamically adjusts attention weights to exploit this, effectively extracting and emphasizing the motion at the right time.

4.  **Benchmark and Validation:** The authors created a new evaluation dataset for this specific task and conducted extensive experiments demonstrating that `FlexiAct` significantly outperforms existing methods in motion accuracy and appearance consistency in these challenging heterogeneous scenarios.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, a novice reader should be familiar with the following concepts:

*   **Diffusion Models:** These are a class of generative models that have become state-of-the-art in image and video generation. They work in two steps:
    1.  **Forward (Noising) Process:** Gradually add random noise to a data sample (like an image) over a series of timesteps until it becomes pure noise.
    2.  **Reverse (Denoising) Process:** Train a neural network (often a U-Net or Transformer) to reverse this process. The network takes a noisy sample and a timestep $t$ as input and predicts the noise that was added at that step. By repeatedly subtracting the predicted noise, the model can generate a clean data sample starting from pure random noise.
        The process is typically optimized by minimizing the mean squared error (MSE) between the actual added noise and the network's predicted noise.

*   **Latent Diffusion Models (LDM):** Training diffusion models directly on high-resolution images or videos is computationally very expensive. LDMs solve this by first using a **Variational Autoencoder (VAE)** to compress the data into a much smaller **latent space**. The diffusion process (both noising and denoising) then occurs entirely within this compact latent space. Once the denoising is complete, a VAE decoder converts the final latent representation back into a full-resolution video. The base model in this paper, `CogVideoX-I2V`, is a latent diffusion model.

*   **Transformers and Attention Mechanism:** The neural network used for denoising in modern diffusion models is often a **Transformer**. The core component of a Transformer is the **attention mechanism**, which allows the model to weigh the importance of different parts of the input when producing an output. In the context of this paper, attention is used for the model to "pay attention" to different conditioning signals, such as the target image or the action information. The standard scaled dot-product attention is calculated as:
    \$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    \$
    - **Q (Query):** Represents the current token that is "looking" for information.
    - **K (Key):** Represents the tokens that are being "looked at." The compatibility between a Query and a Key determines the attention weight.
    - **V (Value):** Represents the actual content of the tokens being looked at.
    - **$d_k$:** The dimension of the Key vectors, used for stabilization.
      The output is a weighted sum of the Value vectors, where the weights are determined by the similarity between Queries and Keys.

*   **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning (PEFT) technique. Instead of fine-tuning all the weights of a large pre-trained model, LoRA freezes the original weights and injects small, trainable "low-rank" matrices into certain layers (like the attention layers of a Transformer). This dramatically reduces the number of trainable parameters, making fine-tuning much faster and more memory-efficient. `RefAdapter` is built using LoRA.

*   **MMDiT (Multi-modal Diffusion Transformer):** The base model `CogVideoX-I2V` uses an `MMDiT` architecture. This is a type of Transformer designed to process multiple types of input (modalities) simultaneously. In a standard Transformer block, all inputs are treated the same. In an `MMDiT`, different inputs (e.g., text prompt embeddings, image latents, noisy video latents) are first concatenated into a single sequence of tokens and then fed into the Transformer blocks. This allows the model to learn complex relationships between the different modalities within a unified architecture.

## 3.2. Previous Works
The paper categorizes related work into three main areas:

*   **Global Motion Customization:** These methods aim to capture the overall "feel" of motion from a reference video and apply it to a new video generation.
    *   **`MotionDirector`**: Uses spatio-temporal LoRA to disentangle appearance and motion from a reference video.
    *   **`Motion Inversion`**: Introduces special embeddings to represent appearance and motion separately.
    *   **Limitation:** These methods are good at creating videos with similar camera motion or general flow but cannot adapt a specific action to a particular subject defined by a target image. The generated video's layout often just mimics the reference video.

*   **Predefined signal-based Action Customization:** This is the most common approach for character animation. It relies on explicit, intermediate representations of motion.
    *   **`ControlNet`**: A general framework for adding conditional control (like pose, depth, edges) to diffusion models. It does this by creating a trainable copy of the model's encoder blocks and feeding the control signal into this copy.
    *   **`AnimateAnyone` / `MagicAnimate`**: These are state-of-the-art methods for human animation. They use a **pose sequence** (skeletal data) extracted from a reference video as the control signal. They also employ a **`ReferenceNet`** (a duplicated network stream) to extract appearance features from a reference image to ensure identity consistency.
    *   **Limitation:** The heavy reliance on pose skeletons makes them brittle. They require strict alignment in structure and viewpoint and are generally not applicable to non-human subjects or abstract characters. `ReferenceNet` also involves duplicating large parts of the network, making it computationally expensive.

*   **Customized Video Generation via Condition Injection:** This refers to general techniques for injecting conditioning information (like an image) into a generation model.
    *   **`IP-Adapter` (Image Prompt Adapter):** A lightweight method that uses a pre-trained image encoder (like CLIP) to generate an image embedding. This embedding is then fed into the cross-attention layers of the diffusion model to guide generation. It's efficient but provides only coarse-grained control, often failing to preserve fine appearance details.
    - **`ReferenceNet`**: As mentioned above, it provides fine-grained control by duplicating network layers to process a reference image, but this comes at a high computational cost due to parameter replication.

## 3.3. Technological Evolution
The field of video animation has evolved rapidly:
1.  **Early GAN-based Methods:** Initial works used Generative Adversarial Networks (GANs) for motion transfer but often suffered from artifacts and temporal instability.
2.  **Emergence of Diffusion Models:** The success of diffusion models in image generation led to their application in video. `ControlNet` was a pivotal work, enabling fine-grained spatial control over image generation, which was quickly extended to video.
3.  **Specialized Animation Models:** Methods like `MagicAnimate` and `AnimateAnyone` refined the `ControlNet` paradigm for human animation, focusing on decoupling pose and appearance for better identity preservation.
4.  **Parallel Track - Global Motion:** Simultaneously, methods like `MotionDirector` explored a different path, focusing on transferring holistic motion patterns without needing a specific character structure.

    `FlexiAct` positions itself as a new direction that bridges the gap. It provides the specific character control of animation models but without their rigid structural constraints, achieving the flexibility needed for heterogeneous scenarios that neither of the previous tracks could handle well.

## 3.4. Differentiation Analysis
Compared to the main prior methods, `FlexiAct` is innovative in several ways:

*   **vs. Pose-based Methods (`AnimateAnyone`):** `FlexiAct` is **signal-free**. It does not require a pose skeleton, depth map, or any other predefined geometric signal. This is its biggest advantage, enabling action transfer between subjects with entirely different shapes, skeletons, and viewpoints (e.g., human to cat).
*   **vs. Global Motion Methods (`MotionDirector`):** `FlexiAct` is **subject-centric**. It animates a specific subject from a target image, adapting the motion to that subject's form. Global motion methods are **scene-centric**; they generate a new scene that has a similar motion pattern to the reference.
*   **vs. `ReferenceNet`:** The proposed `RefAdapter` achieves fine-grained appearance control similar to `ReferenceNet` but is far more **efficient**. It uses LoRA, adding only a small number of trainable parameters (66M, or 5% of the base model), whereas `ReferenceNet` duplicates large portions of the model.
*   **vs. `IP-Adapter`:** `RefAdapter` provides much **finer-grained control** over appearance. `IP-Adapter` uses a single, high-level feature vector from CLIP, which can lose details. `RefAdapter` works with the richer latent representation from the VAE encoder and is trained to preserve consistency.
*   **Novelty of FAE:** The most unique technical contribution is **Frequency-aware Action Extraction (FAE)**. No prior work extracts motion by dynamically modulating attention weights based on the frequency characteristics of different denoising timesteps. This is an elegant, "in-process" approach that avoids the need for a separate, complex motion extraction network.

# 4. Methodology

## 4.1. Principles
The core principle of `FlexiAct` is to **disentangle action from appearance and spatial structure**. To achieve this, the method breaks the problem down into two sub-problems, each handled by a dedicated component:
1.  **Spatial Adaptation and Appearance Preservation:** How to make the generated video feature a subject that looks exactly like the one in the target image, while allowing its pose and position to change according to the desired action? This is handled by **`RefAdapter`**.
2.  **Precise Action Extraction:** How to extract the pure "essence" of an action from a reference video, stripped of its original subject's appearance and layout, so it can be applied to a new subject? This is handled by **`Frequency-aware Action Extraction (FAE)`**.

    The intuition behind `FAE` is particularly novel: the authors observe that the iterative denoising process in diffusion models naturally separates information by frequency. **Early timesteps** (when the image is very noisy) are responsible for establishing coarse, **low-frequency** structures, like the overall motion trajectory. **Later timesteps** (when the image is almost clean) are responsible for refining fine, **high-frequency** details, like texture and appearance. `FAE` exploits this by forcing the model to pay more attention to the action information during the early, low-frequency phase of generation.

## 4.2. Core Methodology In-depth
The overall workflow of `FlexiAct` is shown in Figure 3 of the paper. It involves a two-stage training process followed by a unified inference stage.

![该图像是示意图，展示了FlexiAct方法的工作流程，包括输入图像、动作提取和生成过程。图中各个模块如DiT Block和RefAdapter的功能被标注，强调了频率感知嵌入在动作提取过程中的重要性。](images/3.jpg)
*该图像是示意图，展示了FlexiAct方法的工作流程，包括输入图像、动作提取和生成过程。图中各个模块如DiT Block和RefAdapter的功能被标注，强调了频率感知嵌入在动作提取过程中的重要性。*

### 4.2.1. Base Model: CogVideoX-I2V
`FlexiAct` is built on top of `CogVideoX-I2V`, an `MMDiT`-based latent diffusion model for Image-to-Video generation.
*   **Encoding:** An input image $I$ or video $V$ is first encoded into a latent representation by a VAE encoder ($\epsilon$). A video yields a 4D tensor $L_{video} \in \mathbb{R}^{\frac{T}{4} \times \frac{H}{8} \times \frac{W}{8} \times C}$, and an image yields a 3D tensor $L_{image}$ (with temporal dimension $T=1$).
*   **Conditioning:** In the original `CogVideoX-I2V`, the image latent $L_{image}$ is padded along the temporal dimension and then concatenated with the noisy video latent along the channel dimension. This serves as a strong condition, forcing the generated video to be highly consistent with the image, especially in the first frame.

### 4.2.2. Stage 1: Training RefAdapter
The goal of `RefAdapter` is to teach the model how to preserve the appearance of a subject from a conditioning image while allowing its spatial structure (pose, position) to differ from that image.

*   **Architecture:** `RefAdapter` consists of `LoRA` layers injected into the `MMDiT` blocks of the pre-trained `CogVideoX-I2V`. This makes it lightweight.
*   **Training Process:** The training process is modified from the standard I2V setup in two key ways:
    1.  **Random Frame Conditioning:** Instead of always using the *first frame* of a training video as the condition image, a **random frame is sampled from the video**. This is a crucial step. It creates a discrepancy between the conditioning image (e.g., a person standing mid-way through a video) and the actual start of the video clip being generated. This forces the `RefAdapter` to learn to generate motion that is consistent with the video's action, while ensuring the subject's appearance matches the (potentially differently-posed) conditioning image. It learns to separate appearance from a specific pose.
    2.  **Reference-based Injection:** The latent of the condition image, $L_{image}$, is used to **replace the first temporal embedding** of the video latent $L_{video}$. This is different from the base model's channel-wise concatenation. This change reframes the condition image as a "reference" for appearance that guides the entire generation, rather than a hard constraint that the video's first frame must match.

        Only the LoRA parameters of `RefAdapter` are trained, keeping the base model frozen. This is a one-time training on a large video dataset (`Miradata`).

### 4.2.3. Stage 2: Training Frequency-aware Embedding
After `RefAdapter` is trained, the next stage is to train an embedding that captures the specific action from a given reference video. This is done on a per-video basis.

*   **Architecture:** For each reference video, a new set of learnable parameters, called the **`Frequency-aware Embedding`**, is created. These embeddings are concatenated to the input tokens of the `MMDiT` layers during the denoising process.
*   **Training Process:**
    *   The goal is to make the `Frequency-aware Embedding` memorize both the motion and appearance of the reference video.
    *   The `CogVideoX-I2V` model is fine-tuned to reconstruct the reference video, with only the `Frequency-aware Embedding` parameters being updated.
    *   **Crucially, `RefAdapter` is NOT used during this stage.** This is to prevent `RefAdapter`'s appearance-preserving ability from interfering with the embedding's task of learning the *full* information (motion and appearance) of the reference video.
    *   To prevent the embedding from simply memorizing the video's layout, random cropping is applied to the input video during training.

### 4.2.4. Inference with FAE
This is where all the components come together to perform the final action transfer.

*   **Setup:** The base model, the pre-trained `RefAdapter`, and the newly trained `Frequency-aware Embedding` (for the specific reference action) are all loaded. The input is the target image.
*   **The FAE Insight:** The authors observed that the `Frequency-aware Embedding`, which has learned both motion and appearance, is utilized differently by the model at different denoising timesteps. As shown in Figure 2, at early timesteps (high noise, e.g., $t=800$), the model's attention to this embedding focuses on the moving parts of the subject, capturing low-frequency **motion**. At later timesteps (low noise, e.g., $t=200$), the attention spreads out, capturing high-frequency **appearance details**.
*   **Attention Reweighting Strategy:** To isolate the motion, FAE dynamically increases the attention weights on the `Frequency-aware Embedding` only during the early, motion-focused timesteps. This is achieved by adding a bias, $W_{bias}$, to the attention scores. The formula for the bias is:

    \$
    W _ { b i a s } = \left\{ \begin{array} { l l } { \alpha , } & { t _ { l } \le t \le T } \\ { \displaystyle \frac { \alpha } { 2 } \left[ \cos \left( \displaystyle \frac { \pi } { t _ { h } - t _ { l } } ( t - t _ { l } ) \right) + 1 \right] , } & { t _ { h } \le t < t _ { l } } \\ { 0 , } & { 0 \le t < t _ { h } } \end{array} \right.
    \$

    **Symbol Explanation:**
    - $W_{bias}$: The bias value added to the original attention weight between video tokens and the `Frequency-aware Embedding`.
    - $t$: The current denoising timestep, where $T$ is the total number of steps (e.g., 1000).
    - $\alpha$: A hyperparameter controlling the strength of the bias (set to 1 in practice).
    - $t_l$: The "low-frequency" timestep threshold. For timesteps $t \ge t_l$, the model is considered to be in the motion-focused phase. The paper sets $t_l = 800$.
    - $t_h$: The "high-frequency" timestep threshold. For timesteps $t < t_h$, the model is in the appearance-focused phase. The paper sets $t_h = 700$.

    **Breakdown of the formula:**
    1.  **For $t_l \le t \le T$ (e.g., timesteps 800 to 1000):** This is the earliest phase of denoising. The full bias $\alpha$ is applied. This forces the model to strongly "listen" to the `Frequency-aware Embedding` to establish the correct motion dynamics.
    2.  **For $t_h \le t < t_l$ (e.g., timesteps 700 to 799):** This is a transition phase. A cosine function smoothly decays the bias from $\alpha$ down to 0. This smooth transition is critical to prevent artifacts that would arise from an abrupt change in guidance.
    3.  **For $0 \le t < t_h$ (e.g., timesteps 0 to 699):** This is the late phase of denoising. The bias is zero. The model now ignores the action embedding's appearance information and relies on `RefAdapter` and the target image condition to generate the correct appearance and fine details.

        The final attention weight is $W_{attn} = W_{ori} + W_{bias}$. By applying this dynamic reweighting, `FAE` effectively "distills" the motion from the reference video and applies it to the target subject, whose appearance is simultaneously maintained by `RefAdapter`.

# 5. Experimental Setup

## 5.1. Datasets
*   **Training Dataset:** For training the `RefAdapter`, the authors used **`Miradata`**, a large-scale video dataset containing 42,000 videos with long durations and structured captions. The diversity and scale of this dataset are crucial for training a robust and generalizable adapter.
*   **Evaluation Dataset:** The authors constructed a custom evaluation dataset to specifically test performance in heterogeneous scenarios. It consists of **250 video-image pairs**, comprising **25 distinct action categories** (e.g., yoga, jumping, running) and **10 different target images for each action**. The target images are highly diverse, including real humans, animals, and characters from animations and games. This diversity is essential for comprehensively evaluating the method's generalization capabilities.

## 5.2. Evaluation Metrics
The paper uses a combination of automatic metrics and human evaluation.

*   **Text Similarity:**
    1.  **Conceptual Definition:** This metric measures how well the generated video content aligns with a given text prompt. A higher score indicates better semantic consistency.
    2.  **Mathematical Formula:** It is calculated as the cosine similarity between the CLIP embeddings of the video frames and the CLIP embedding of the text prompt. For a video with $T$ frames, the average similarity is often used.
        \$
        \text{Text Similarity} = \frac{1}{T} \sum_{i=1}^{T} \frac{E_{frame_i} \cdot E_{text}}{\|E_{frame_i}\| \|E_{text}\|}
        \$
    3.  **Symbol Explanation:**
        - $E_{frame_i}$: The CLIP image feature vector for the $i$-th frame of the generated video.
        - $E_{text}$: The CLIP text feature vector for the input prompt.

*   **Motion Fidelity:**
    1.  **Conceptual Definition:** This metric quantifies how closely the motion in the generated video matches the motion in the reference video, even if the subjects and layouts are different.
    2.  **Mathematical Formula:** The paper follows `Yatim et al. 2023` and uses a tracking model (`CoTracker`) to extract motion trajectories (tracklets) from both the reference and generated videos. The similarity between these sets of trajectories is then computed. While the exact formula is not in this paper, it typically involves comparing properties of the trajectories, such as displacement vectors or temporal velocity profiles.
    3.  **Symbol Explanation:** This metric relies on the output of an external model (`CoTracker`) to represent motion, which is then compared using a similarity metric.

*   **Temporal Consistency:**
    1.  **Conceptual Definition:** This measures the smoothness and coherence of the video, checking for flickering or sudden, unnatural changes between consecutive frames. Higher values indicate a more stable video.
    2.  **Mathematical Formula:** It is calculated as the average CLIP similarity between all pairs of frames in the generated video.
        \$
        \text{Temporal Consistency} = \frac{2}{T(T-1)} \sum_{i=1}^{T-1} \sum_{j=i+1}^{T} \frac{E_{frame_i} \cdot E_{frame_j}}{\|E_{frame_i}\| \|E_{frame_j}\|}
        \$
    3.  **Symbol Explanation:**
        - $E_{frame_i}$ and $E_{frame_j}$: The CLIP image feature vectors for frames $i$ and $j$.
        - $T$: The total number of frames in the video.

*   **Appearance Consistency:**
    1.  **Conceptual Definition:** This metric assesses how well the appearance of the subject in the generated video matches the appearance of the subject in the target image.
    2.  **Mathematical Formula:** The paper defines this as the average CLIP similarity between the *first frame* and the *remaining frames* of the output video.
        \$
        \text{Appearance Consistency} = \frac{1}{T-1} \sum_{i=2}^{T} \frac{E_{frame_1} \cdot E_{frame_i}}{\|E_{frame_1}\| \|E_{frame_i}\|}
        \$
    3.  **Symbol Explanation:**
        - $E_{frame_1}$ and $E_{frame_i}$: The CLIP image feature vectors for the first frame and the $i$-th frame, respectively.
        - *Note on this metric:* As the authors aim to loosen the first-frame constraint, a more direct metric would be to compare every generated frame with the original target image: $avg(CLIP_sim(target_image, generated_frame_i))$. However, the chosen metric still serves as a proxy for internal consistency.

*   **Human Evaluation:** 5 human raters were asked to compare pairs of videos (one from a baseline, one from another method) and choose which one was better in terms of **Motion Consistency** (with the reference video) and **Appearance Consistency** (with the target image).

## 5.3. Baselines
*   **MD-I2V:** The authors' reimplementation of `MotionDirector` on the same `CogVideoX-I2V` backbone. This ensures a fair comparison by using a stronger and identical base model for the baseline.
*   **BaseModel:** A simplified version of `FlexiAct` that uses standard learnable embeddings for action but **without `RefAdapter` and `FAE`**. This serves as a strong ablation baseline to demonstrate the effectiveness of the two proposed components.
*   Pose-based methods were excluded from the main comparison because they are fundamentally unsuitable for the "heterogeneous scenarios" targeted by the paper, as demonstrated in Figure 4.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results strongly validate the effectiveness of `FlexiAct`.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">Automatic Evaluations</th>
<th colspan="3">Human Evaluations</th>
</tr>
<tr>
<th>Text Similarity ↑</th>
<th>Motion Fidelity ↑</th>
<th>Temporal Consistency ↑</th>
<th>Appearance Consistency ↑</th>
<th></th>
<th>Motion Consistency</th>
<th>Appearance Consistency</th>
</tr>
</thead>
<tbody>
<tr>
<td>MD-I2V [Zhao et al. 2023]</td>
<td>0.2446</td>
<td>0.3496</td>
<td>0.9276</td>
<td>0.8963</td>
<td>v.s. Base Model</td>
<td>47.2 v.s. 52.8</td>
<td>53.1 v.s. 46.9</td>
</tr>
<tr>
<td>Base Model</td>
<td>0.2541</td>
<td>0.3562</td>
<td>0.9283</td>
<td>0.8951</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o FAE</td>
<td>0.2675</td>
<td>0.3614</td>
<td>0.9255</td>
<td>0.9134</td>
<td>v.s. Base Model</td>
<td>59.7 v.s. 40.3</td>
<td>76.4 v.s. 23.6</td>
</tr>
<tr>
<td>w/o RefAdapter</td>
<td>0.2640</td>
<td>0.3856</td>
<td>0.9217</td>
<td>0.9021</td>
<td>v.s. Base Model</td>
<td>68.6 v.s. 31.4</td>
<td>52.2 v.s. 47.8</td>
</tr>
<tr>
<td>Ours</td>
<td><strong>0.2732</strong></td>
<td><strong>0.4103</strong></td>
<td><strong>0.9342</strong></td>
<td><strong>0.9162</strong></td>
<td>v.s. Base Model</td>
<td><strong>79.5</strong> v.s. 20.5</td>
<td><strong>78.3</strong> v.s. 21.7</td>
</tr>
</tbody>
</table>

**Analysis of Quantitative Results:**
*   **`Ours` (FlexiAct) vs. Baselines:** `FlexiAct` achieves the highest scores across all four automatic metrics. The most significant improvements are in **`Motion Fidelity`** (0.4103 vs. 0.3496 for MD-I2V and 0.3562 for BaseModel) and **`Appearance Consistency`** (0.9162 vs. 0.8963 for MD-I2V). This directly supports the paper's central claim: `FlexiAct` is superior at both accurately transferring motion and preserving the target's appearance.
*   **`MD-I2V` Performance:** The global motion method `MD-I2V` scores poorly on `Motion Fidelity` and `Appearance Consistency`, confirming that it struggles to adapt a specific action to a target character.
*   **Human Evaluation:** The human preference scores are even more decisive. `FlexiAct` is overwhelmingly preferred over the `BaseModel` for both motion (79.5% preference) and appearance (78.3% preference), indicating its generated videos are qualitatively much better.

**Qualitative Analysis:**
As shown in Figure 5, `MD-I2V` and `BaseModel` exhibit clear flaws. For instance, they may fail to complete an action correctly (man not standing up) or introduce severe appearance artifacts (clothing appearing on the subject). In contrast, `FlexiAct` produces videos that are both accurate to the reference motion and consistent with the target image's appearance. The various examples in Figures 7-10 further demonstrate its robustness across diverse subjects and domains, including human-to-animal action transfer, which is a particularly challenging task that most other methods cannot handle at all.

![该图像是一个示意图，展示了动作定制的方法对比。第一行是参考视频，第二行是MD-12V模型结果，第三行是基础模型结果，第四行则是我们的方法效果。该图像展示了不同模型在动作转移任务中的表现，突出我们方法的优越性。](images/5.jpg)
*该图像是一个示意图，展示了动作定制的方法对比。第一行是参考视频，第二行是MD-12V模型结果，第三行是基础模型结果，第四行则是我们的方法效果。该图像展示了不同模型在动作转移任务中的表现，突出我们方法的优越性。*

## 6.2. Ablation Studies / Parameter Analysis
The ablation studies in Table 1 and Figure 6 systematically validate the contribution of each proposed component.

![该图像是示意图，展示了使用不同方法进行动作转移的效果对比。第一行是参考视频，接下来的行分别展示了使用和不使用FAE及RefAda的效果，可以观察到动作传递的变化和相应的适应性。](images/6.jpg)
*该图像是示意图，展示了使用不同方法进行动作转移的效果对比。第一行是参考视频，接下来的行分别展示了使用和不使用FAE及RefAda的效果，可以观察到动作传递的变化和相应的适应性。*

*   **Effectiveness of FAE:** When `FAE` is removed (`w/o FAE`), the `Motion Fidelity` score drops from 0.4103 to 0.3614. This is a significant decrease, confirming that the `FAE` attention reweighting strategy is crucial for accurately extracting and applying the action. The qualitative results in Figure 6 show that without `FAE`, the character's movements are incorrect and do not match the reference video (e.g., just raising a hand instead of stretching).

*   **Effectiveness of RefAdapter:** When `RefAdapter` is removed (`w/o RefAdapter`), both `Appearance Consistency` (0.9162 to 0.9021) and `Motion Fidelity` (0.4103 to 0.3856) decrease. The drop in appearance consistency is expected, as `RefAdapter`'s main job is to preserve appearance. The drop in motion fidelity is also logical: without `RefAdapter`'s ability to handle spatial adaptation, the model struggles to apply the motion correctly to a structurally different target, leading to distorted actions. Figure 6 confirms this, showing that without `RefAdapter`, facial details and clothing are inconsistent, and movements are not fully executed.

*   **Analysis of FAE's Bias Transition:** Figure 11 provides a critical analysis of the smooth transition function in the `FAE` formula. It shows that an abrupt change in the attention bias leads to poor results.
    *   If the bias is turned off too early (e.g., at $t=800$), the motion is inaccurate because the model doesn't get enough motion guidance.
    *   If the bias is kept on for too long (e.g., turned off at $t=700$), the model starts picking up unwanted *appearance* details from the reference video (like the reference subject's clothing).
    *   The smooth cosine transition between $t=800$ and $t=700$ is therefore essential for achieving a clean separation of motion and appearance.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper introduces `FlexiAct`, a novel and powerful framework for action transfer in video generation. It successfully addresses a major limitation of prior work by enabling action transfer to subjects in "heterogeneous scenarios," where the target subject may have a completely different structure, viewpoint, or layout from the source. The key contributions are twofold:
1.  **`RefAdapter`**, a lightweight adapter that provides robust appearance consistency and spatial adaptation.
2.  **`Frequency-aware Action Extraction (FAE)`**, an innovative, in-process technique that leverages the frequency-separating nature of the diffusion denoising process to precisely extract motion without a separate network.

    Through extensive experiments, the authors demonstrate that `FlexiAct` significantly outperforms existing methods in both motion accuracy and appearance preservation, opening up new possibilities for flexible and creative character animation.

## 7.2. Limitations & Future Work
The authors acknowledge one primary limitation:
*   **Per-Video Optimization:** To transfer an action from a new reference video, `FlexiAct` requires training a new `Frequency-aware Embedding` for that specific video. This process, while relatively short (1,500-3,000 steps), is still a form of per-instance optimization and not a "zero-shot" or "feed-forward" process. This limits its scalability for applications requiring rapid transfer from arbitrary videos on the fly.

    As a direction for **future work**, they suggest developing a feed-forward motion transfer method for heterogeneous scenarios that would not require this per-video training step.

## 7.3. Personal Insights & Critique
*   **Key Innovation:** The standout contribution of this paper is unquestionably **`FAE`**. The conceptual link between denoising timesteps and information frequency (motion vs. detail) is a profound insight. The implementation via dynamic attention modulation is both elegant and effective. This idea of controlling the information flow during generation based on the timestep has significant potential beyond action transfer. It could be adapted for tasks like style transfer (transferring structure first, then texture) or guided editing (making coarse changes early and fine-tuning later).

*   **Strengths:**
    *   **Problem Formulation:** The paper clearly identifies a practical and important problem (heterogeneous action transfer) that was largely unaddressed by previous research.
    *   **Elegant Solution:** The `FlexiAct` framework is well-designed, with each component having a clear and distinct purpose. The combination of `RefAdapter` for appearance and `FAE` for motion is a powerful pairing.
    *   **Strong Empirical Evidence:** The experiments are comprehensive, with a well-chosen custom benchmark, strong baselines, and insightful ablation studies that convincingly support the authors' claims.

*   **Potential Issues and Areas for Improvement:**
    *   **Scalability:** The per-video optimization is a significant practical hurdle, as the authors noted. A future model that could encode motion from any video in a feed-forward manner would be a major breakthrough.
    *   **Complexity of Actions:** The actions shown in the paper are mostly cyclic and self-contained (e.g., dancing, exercising). It's unclear how well `FlexiAct` would handle more complex, non-repetitive, or interactive actions (e.g., a person picking up an object and handing it to another person). The "action" is encoded in a fixed-size embedding, which might struggle to capture very long or intricate temporal dependencies.
    *   **Evaluation Metric for Appearance:** The paper's `Appearance Consistency` metric is slightly weak, as it measures internal consistency rather than fidelity to the original target image. While the qualitative results are strong, a more direct metric would strengthen the quantitative evaluation.

        Overall, `FlexiAct` is a high-quality research paper that makes a significant and innovative contribution to the field of controllable video generation. Its core ideas, especially `FAE`, are likely to inspire future research in fine-grained control of generative models.