# 1. Bibliographic Information

## 1.1. Title
StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text

## 1.2. Authors
Roberto Henschel$^{1*\dagger}$, Levon Khachatryan$^{1*}$, Hayk Poghosyan$^{1*}$, Daniil Hayrapetyan$^1$, Vahram Tadevosyan$^{1\ddagger}$, Zhangyang Wang$^{1,2}$, Shant Navasardyan$^1$, Humphrey Shi$^{1,3}$

*   **Affiliations:**
    1.  Picsart AI Research (PAIR)
    2.  UT Austin
    3.  Georgia Tech
*   **Notes:** `*` indicates equal contribution. $\dagger$ and $\ddagger$ indicate project leads or corresponding authors.

## 1.3. Journal/Conference
*   **Published at (UTC):** 2024-03-21
*   **Venue:** This paper appears to be a preprint published on **arXiv** (arXiv:2403.14773). It is common for cutting-edge generative AI research to be disseminated via arXiv first. The paper mentions "Published at (UTC): 2024-03-21T18:27:29.000Z".

## 1.4. Publication Year
2024

## 1.5. Abstract
The paper addresses the limitation of existing text-to-video (T2V) diffusion models, which excel at short videos (typically 16-24 frames) but fail to generate consistent long videos, often resulting in hard cuts or "stagnation" (lack of motion). The authors propose **StreamingT2V**, an autoregressive framework capable of generating videos of 1200+ frames (2 minutes or more).

The core technical contributions are:
1.  **Conditional Attention Module (CAM):** A short-term memory block using attention to ensure smooth transitions between video chunks.
2.  **Appearance Preservation Module (APM):** A long-term memory block that extracts features from the first video chunk to prevent "forgetting" the initial scene or object identity.
3.  **Randomized Blending:** A technique to apply high-resolution video enhancers autoregressively without creating visible seams between chunks.

## 1.6. Original Source Link
*   **PDF Link:** [https://arxiv.org/pdf/2403.14773v2.pdf](https://arxiv.org/pdf/2403.14773v2.pdf)
*   **Project/Code:** [https://github.com/Picsart-AI-Research/StreamingT2V](https://github.com/Picsart-AI-Research/StreamingT2V)

# 2. Executive Summary

## 2.1. Background & Motivation
**The Core Problem:**
Text-to-video (T2V) generation has seen rapid advancements with diffusion models. However, most state-of-the-art models (like ModelScope, SVD) are trained to generate very short clips (e.g., 2-4 seconds).
*   **Naive Extension Fails:** Simply asking the model to generate the "next" chunk of video often results in **hard cuts** (the scene changes abruptly) or **inconsistency** (the object changes color or shape).
*   **Autoregressive Stagnation:** Existing methods that try to extend videos by conditioning on the last frame of the previous chunk often suffer from **video stagnation**. The model becomes "lazy," producing static frames where the background freezes and motion stops, effectively turning the video into a still image.
*   **Forgetting:** As generation proceeds, the model "forgets" the original subject's appearance (e.g., a person's clothes change style over 20 seconds).

**Innovation Point:**
StreamingT2V proposes that simple concatenation or standard conditioning (like adding the previous frame to the input) is insufficient. Instead, it introduces specific **memory modules** (short-term and long-term) that inject information into the diffusion process via **attention mechanisms**, allowing for high motion preservation and identity consistency over long durations.

## 2.2. Main Contributions & Findings
1.  **StreamingT2V Framework:** A pipeline that separates generation into three stages: Initialization (first chunk), Streaming T2V (autoregressive extension), and Streaming Refinement (upscaling).
2.  **Conditional Attention Module (CAM):** A novel architecture inspired by ControlNet but using attention. It allows the model to "look back" at the previous chunk's features to maintain motion continuity without being rigidly locked to the previous frame's pixel structure.
3.  **Appearance Preservation Module (APM):** A mechanism that anchors the generation to the very first frame, ensuring that the global scene and object identity remain consistent even after hundreds of frames.
4.  **Randomized Blending for Enhancement:** A stochastic method for merging overlapping video chunks during the upscaling/refinement phase, eliminating the "seams" that typically appear when enhancing a video piece-by-piece.
5.  **Superior Performance:** Experiments show StreamingT2V achieves the highest motion consistency (lowest Motion Aware Warp Error) and text alignment compared to competitors like I2VGen-XL, SVD, and SparseCtrl, which suffer from stagnation.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, a beginner needs to grasp several key concepts in generative AI:

*   **Diffusion Models:** These are generative models that learn to create data (images or videos) by reversing a noise process. They start with random noise and iteratively refine it into a clear image/video based on learned patterns.
    *   **Latent Diffusion Models (LDM):** Instead of operating on raw pixels (which is computationally expensive), these models operate in a compressed "latent space" (a mathematical representation of the image content).
*   **Autoregressive Generation:** A process where a sequence is generated piece-by-piece. For video, the model generates chunk 1, then uses chunk 1 to help generate chunk 2, and so on. The challenge is preventing errors from accumulating over time.
*   **UNet:** The backbone neural network architecture used in most diffusion models. It has an encoder (downsampling) and a decoder (upsampling) with "skip connections" that preserve fine details.
*   **Attention Mechanism:** A mathematical operation that allows the model to weigh the importance of different parts of the input.
    *   **Self-Attention:** The model looks at different parts of the *current* image/video frame to understand context.
    *   **Cross-Attention:** The model looks at an *external* signal (like a text prompt or a reference image) to guide the generation.
*   **CLIP (Contrastive Language-Image Pre-training):** A model trained to understand the relationship between text and images. It converts text (prompts) and images into mathematical vectors (embeddings). If the vectors are close, the text and image are semantically similar.

## 3.2. Previous Works
The paper builds upon and contrasts itself with several existing approaches:

*   **Short Video Generators (The Base):** Models like **ModelScope**, **SVD (Stable Video Diffusion)**, and **I2VGen-XL** generate high-quality but short clips. StreamingT2V uses ModelScope as its base building block.
*   **Image-to-Video Adapters:** Methods like **SparseCtrl**, **SEINE**, and **DynamiCrafter** attempt to animate a starting image.
    *   *Limitation:* When used autoregressively (using the last frame of chunk $N$ as the input for chunk $N+1$), these models often fail. The paper argues they use "weak" conditioning (like simple concatenation or CLIP embeddings), which leads to quality degradation.
*   **Approaches with Strong Priors:**
    *   **FreeNoise:** Re-uses the noise vectors to force consistency. *Critique:* This restricts motion too much, leading to static videos.
    *   **Gen-L:** Blends overlapping videos. *Critique:* Can cause stagnation.

## 3.3. Differentiation Analysis
StreamingT2V distinguishes itself through its specific memory architecture:
*   **vs. SparseCtrl/ControlNet:** SparseCtrl adds zero-filled frames to match input sizes, causing input inconsistency. StreamingT2V's **CAM** uses attention, which handles the transition more naturally without needing "dummy" zero inputs.
*   **vs. SVD/I2VGen-XL:** These rely heavily on CLIP image embeddings of the conditioning frame. The authors argue CLIP loses spatial detail needed for seamless transitions. StreamingT2V injects features directly into the UNet via **CAM**.
*   **vs. FreeNoise:** FreeNoise achieves consistency by sacrificing motion. StreamingT2V achieves consistency *while preserving high motion* (dynamic content).

# 4. Methodology

## 4.1. Principles
The core philosophy of StreamingT2V is that generating a long video requires two types of memory:
1.  **Short-Term Memory:** "What just happened?" (To ensure smooth motion from the previous second). Handled by **CAM**.
2.  **Long-Term Memory:** "Who is in the video and where are they?" (To ensure the person doesn't change into someone else). Handled by **APM**.

    The overall pipeline consists of three stages:
1.  **Initialization:** Generate the first 16 frames ($256 \times 256$).
2.  **Streaming T2V:** Autoregressively generate subsequent 240+ frames ($256 \times 256$).
3.  **Streaming Refinement:** Upscale the entire video to $720 \times 720$ using a Refiner model with Randomized Blending.

    The following figure (Figure 2 from the original paper) illustrates this high-level pipeline. Note the separation into the initialization, streaming, and refinement stages:

    ![该图像是一个示意图，展示了StreamingT2V方法的三个阶段：初始化阶段、Streaming T2V阶段和Streaming Refinement阶段。初始化阶段使用预训练模型生成初始特征，通过条件注意模块（CAM）和外观保持模块（APM）进行短期和长期的特征提取，确保生成视频的一致性与流畅度。](images/2.jpg)
    *该图像是一个示意图，展示了StreamingT2V方法的三个阶段：初始化阶段、Streaming T2V阶段和Streaming Refinement阶段。初始化阶段使用预训练模型生成初始特征，通过条件注意模块（CAM）和外观保持模块（APM）进行短期和长期的特征提取，确保生成视频的一致性与流畅度。*

## 4.2. Core Methodology In-depth

### 4.2.1. The Conditional Attention Module (CAM)
**Purpose:** This is the short-term memory. It conditions the generation of the *current* video chunk on the *last 8 frames* of the *previous* chunk.

**Mechanism:**
CAM acts as a "sidecar" network attached to the main Video-LDM (Latent Diffusion Model). It extracts features from the conditioning frames and injects them into the main model.

**Step-by-Step Process & Formulas:**

1.  **Feature Extraction:**
    A frame-wise image encoder $\mathcal{E}_{\mathrm{cond}}$ processes the conditioning frames (the last $F_{\mathrm{cond}} = 8$ frames of the previous chunk).
    These features pass through a copy of the Video-LDM's encoder layers (up to the middle block). Let the output of the CAM feature extractor be $x_{\mathrm{CAM}}$.
    $x_{\mathrm{CAM}} \in \mathbb{R}^{(b \cdot w \cdot h) \times F_{\mathrm{cond}} \times c}$, where $b$ is batch size, `w, h` are spatial dimensions, and $c$ is channels.

2.  **Feature Preparation:**
    The main Video-LDM UNet has "skip connections" that carry information from its encoder to its decoder. Let a feature map in a skip connection be $x_{\mathrm{SC}} \in \mathbb{R}^{b \times F \times h \times w \times c}$.
    The model applies a linear map $P_{\mathrm{in}}$ to $x_{\mathrm{SC}}$ to prepare it for attention, resulting in $x_{\mathrm{SC}}'$.

3.  **Temporal Multi-Head Attention (T-MHA):**
    This is the critical step. The current generation ($x_{\mathrm{SC}}'$) "attends" to the memory ($x_{\mathrm{CAM}}$).
    The attention is calculated using Queries ($Q$), Keys ($K$), and Values ($V$).
    *   **Queries ($Q$):** Derived from the *current* generation features ($x_{\mathrm{SC}}'$).
    *   **Keys ($K$) & Values ($V$):** Derived from the *previous* chunk's features ($x_{\mathrm{CAM}}$).

        The projection formulas using learnable linear maps $P_Q, P_K, P_V$ are:
    `Q = P_Q(x_{\mathrm{SC}}')`
    `K = P_K(x_{\mathrm{CAM}})`
    `V = P_V(x_{\mathrm{CAM}})`

    The attention operation fuses these:
    $x_{\mathrm{SC}}'' = \mathrm{T\text{-}MHA}(Q, K, V)$

    **Why this matters:** By using attention, the model finds semantic correspondences between the new frames (Queries) and the old frames (Keys/Values). It doesn't just copy pixels; it understands "the object at position X moved to position Y".

4.  **Feature Injection:**
    The attention output $x_{\mathrm{SC}}''$ is processed by an output linear map $P_{\mathrm{out}}$ and a zero-convolution layer (represented by $R$). It is then added back to the original skip connection features.

    $x_{\mathrm{SC}}''' = x_{\mathrm{SC}} + R(P_{\mathrm{out}}(x_{\mathrm{SC}}''))$

    *   $x_{\mathrm{SC}}'''$: The final feature map used by the main Video-LDM.
    *   $R$: A zero-initialized convolution. This ensures that at the start of training, the CAM module outputs zeros, effectively doing nothing. This allows the model to start with its original pre-trained behavior and gradually learn to use the memory.

        The following figure (Figure 3 from the original paper) visualizes the CAM and APM architectures. Notice the "Feature Injector" in CAM using Cross-Attention, and the APM modifying the Cross-Attention inputs:

        ![该图像是StreamingT2V论文中的示意图，展示了Conditional Attention Module和Appearance Preservation Module的结构。图中表示了不同模块之间的数据流动和处理方式，明确地显示了如何将条件注意力机制应用于视频生成的过程。](images/3.jpg)
        *该图像是StreamingT2V论文中的示意图，展示了Conditional Attention Module和Appearance Preservation Module的结构。图中表示了不同模块之间的数据流动和处理方式，明确地显示了如何将条件注意力机制应用于视频生成的过程。*

### 4.2.2. The Appearance Preservation Module (APM)
**Purpose:** This is the long-term memory. It prevents the video from drifting away from the initial concept (e.g., ensuring a "man in a red shirt" doesn't become a "man in a blue shirt" 10 seconds later).

**Mechanism:**
APM modifies the standard Cross-Attention layers of the UNet. Normally, these layers only attend to the text prompt. APM forces them to also attend to an **anchor frame** (the very first frame of the video).

**Step-by-Step Process & Formulas:**

1.  **Token Preparation:**
    *   **Text Tokens ($x_{\mathrm{text}}$):** Standard CLIP text encoding of the prompt.
    *   **Image Tokens ($x_{\mathrm{mixed}}$):** The CLIP image embedding of the anchor frame is expanded into $k=16$ tokens using a Multi-Layer Perceptron (MLP). These represent the visual "essence" of the start of the video.

2.  **Weighted Mixing:**
    The model needs to balance "following the text" and "keeping the appearance." The paper introduces a learnable parameter $\alpha_l$ for each layer $l$.
    The input to the cross-attention mechanism ($x_{\mathrm{cross}}$) becomes a weighted sum of the image tokens and text tokens:

    $x_{\mathrm{cross}} = \mathrm{SiLU}(\alpha_l)x_{\mathrm{mixed}} + x_{\mathrm{text}}$

    *   $x_{\mathrm{cross}}$: The combined context vector used as Keys/Values in the UNet's cross-attention.
    *   $\alpha_l$: A learnable scalar initialized to 0.
    *   $\mathrm{SiLU}$: The Sigmoid Linear Unit activation function ($\text{SiLU}(x) = x \cdot \sigma(x)$).
    *   $x_{\mathrm{mixed}}$: The anchor frame features (long-term memory).
    *   $x_{\mathrm{text}}$: The prompt features.

        **Analysis:** Since $\alpha_l$ is initialized to 0, $\text{SiLU}(0) = 0$. This means initially, $x_{\mathrm{cross}} = x_{\mathrm{text}}$, so the model behaves like a standard text-to-video model. As training progresses, $\alpha_l$ grows, and the model learns to incorporate the anchor frame information ($x_{\mathrm{mixed}}$).

### 4.2.3. Auto-regressive Video Enhancement (Refinement Stage)
**Purpose:** The base generation is low resolution ($256 \times 256$). We need to upscale it to $720 \times 720$. A separate high-res model (Refiner) is used.
**Challenge:** If we upscale chunk-by-chunk, the edges of the chunks won't match, creating "seams" or "flicker" every 24 frames.

**Solution: Randomized Blending.**
The authors process overlapping chunks (overlap $\mathcal{O} = 8$ frames). They blend the *latent noise representations* of these chunks stochastically.

**Step-by-Step Process & Formulas:**

1.  **Shared Noise Generation:**
    When adding noise to the video chunks for the SDEdit process (refining), the noise must be consistent in the overlapping regions.
    For a chunk $i$, the noise $\epsilon_i$ is formed by concatenating the noise from the *previous* chunk's overlap ($\epsilon_{i-1}^{(F-\mathcal{O}):F}$) and new random noise ($\hat{\epsilon}_i$):

    $\epsilon_i := \mathrm{concat}([\epsilon_{i-1}^{(F-\mathcal{O}):F}, \hat{\epsilon}_i], \dim=0)$

2.  **Randomized Latent Blending:**
    During the denoising steps, we have two versions of the overlapping frames:
    *   $x_L$: The end of chunk `i-1` (smooth transition from the past).
    *   $x_R$: The start of chunk $i$ (smooth transition to the future).

        Instead of blending them with a fixed average, the authors randomly pick a "cut point" $f_{\mathrm{thr}}$ within the overlap region $\{0, \ldots, \mathcal{O}\}$.
    They construct a merged latent $x_{LR}$ by taking the left part from $x_L$ and the right part from $x_R$:

    $x_{LR} := \mathrm{concat}([x_L^{1:F-f_{\mathrm{thr}}}, x_R^{f_{\mathrm{thr}}+1:F}], \mathrm{dim}=0)$

    Then, for any frame $f$ in the overlap, the final latent is chosen from chunk `i-1` with probability:
    $P(\text{choose } L) = 1 - \frac{f}{\mathcal{O} + 1}$

    **Intuition:** By randomly shifting the cut point at every single denoising step (of which there are many), the boundary effectively "blurs" out. The model never sees a hard line, so it generates a seamless transition.

The following figure (Figure 4 from the original paper) demonstrates the impact of this blending. Notice how naive concatenation (a) creates cuts, shared noise (b) helps but still has misalignment, and randomized blending (c) is seamless:

![该图像是一个示意图，展示了三种视频生成方法的对比：左侧是简单拼接方法（Naive Concatenation），中间是共享噪声方法（Shared Noise），右侧是随机混合方法（Randomized Blending）。图中以横向时间切片（XT Slice）和视频帧对比，显示了不同方法在生成长视频时的效果差异。](images/4.jpg)
*该图像是一个示意图，展示了三种视频生成方法的对比：左侧是简单拼接方法（Naive Concatenation），中间是共享噪声方法（Shared Noise），右侧是随机混合方法（Randomized Blending）。图中以横向时间切片（XT Slice）和视频帧对比，显示了不同方法在生成长视频时的效果差异。*

# 5. Experimental Setup

## 5.1. Datasets
*   **Training Data:** The paper uses an **internal dataset** of videos. Specific details (size, content) are not disclosed in the snippet, which is common for industry labs (Picsart).
*   **Test Set:** To evaluate performance, the authors created a benchmark of **50 prompts** covering diverse categories:
    *   Actions (e.g., "A knight riding on a horse")
    *   Objects (e.g., "A steampunk robot")
    *   Scenes (e.g., "Drone flythrough of a tropical jungle")
    *   Camera Motions (e.g., "Camera is zooming out")
    *   **Sample Prompt:** "A squirrel in Antarctica, on a pile of hazelnuts."

## 5.2. Evaluation Metrics
The authors use three primary metrics to assess quality, consistency, and text alignment.

### 5.2.1. SCuts (Scene Cuts)
*   **Conceptual Definition:** This metric detects abrupt changes in the video that look like video editing cuts. In a continuous generated video, there should be *zero* scene cuts. A high score means the model is glitching between chunks.
*   **Implementation:** Uses the `AdaptiveDetector` from the `PySceneDetect` library.

### 5.2.2. CLIP Score (Text Alignment)
*   **Conceptual Definition:** Measures how well the video frames match the text prompt.
*   **Formula:** It computes the cosine similarity between the text embedding and the image embedding of the frames.
    `CLIP(\mathcal{V}, \text{text}) = \frac{1}{F} \sum_{f=1}^{F} \cos(\mathcal{E}_{txt}(\text{text}), \mathcal{E}_{img}(v_f))`
    *(Note: While the exact formula isn't in the text, this is the standard definition implied by "cosine similarity from the CLIP text encoding to the CLIP image encodings".)*

### 5.2.3. Motion Aware Warp Error (MAWE)
*   **Conceptual Definition:** This is a novel metric proposed by the authors. Standard metrics can be "gamed"—a static video has perfect consistency but zero motion. MAWE penalizes both **inconsistency** (high warp error) and **stagnation** (low motion). A *lower* MAWE is better.
*   **Mathematical Formula:**
    $ \mathrm{MAWE}(\mathcal{V}) := \frac{W(\mathcal{V})}{\mathrm{OFS}(\mathcal{V})} $
*   **Symbol Explanation:**
    *   $W(\mathcal{V})$: **Mean Warp Error**. Measures the pixel distance between a frame and the next frame *after* warping it based on optical flow. High error = inconsistency/artifacts.
    *   $\mathrm{OFS}(\mathcal{V})$: **Optical Flow Score**. The mean squared magnitude of optical flow vectors. High OFS = high motion.
    *   **Logic:** If a video is static, $OFS \approx 0$, so MAWE $\to \infty$ (bad). If a video is inconsistent, $W$ is high, so MAWE is high (bad). Only a video with high motion (high denominator) and high consistency (low numerator) gets a low MAWE score.

## 5.3. Baselines
The method is compared against several state-of-the-art open-source models adapted for long generation:
1.  **SparseCtrl:** Uses a ControlNet-like sparse encoder.
2.  **I2VGen-XL:** A cascaded diffusion model from Alibaba.
3.  **DynamiCrafter-XL:** Animates images using video priors.
4.  **SEINE:** Focuses on transitions.
5.  **SVD (Stable Video Diffusion):** Stability AI's model.
6.  **FreeNoise:** A training-free method enforcing noise consistency.
7.  **OpenSora / OpenSoraPlan:** Transformer-based diffusion models.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experiments demonstrate that StreamingT2V outperforms competitors in generating long, high-motion, consistent videos.

**Key Findings:**
1.  **Motion vs. Stagnation:** Most baselines (like FreeNoise, I2VGen-XL) produce "video stagnation," where the video effectively pauses or moves very little to maintain consistency. This results in high (bad) MAWE scores or deceptively low SCuts scores (because a static image has no cuts).
2.  **Consistency:** StreamingT2V achieves a very low **SCuts** score (0.04), comparable to FreeNoise (0), but with significantly higher motion. Models like SparseCtrl fail dramatically here (SCuts 5.48), creating a jarring "strobe light" effect due to input padding inconsistencies.
3.  **Text Alignment:** StreamingT2V achieves the highest **CLIP score**, indicating that the **APM** successfully preserves the semantic content (the "what") throughout the long video, whereas other models drift or degrade.

## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper. Note that **lower is better** for MAWE and SCuts, while **higher is better** for CLIP.

| Method | ↓MAWE | ↓SCuts | ↑CLIP |
| :--- | :--- | :--- | :--- |
| SparseCtrl [11] | 6069.7 | 5.48 | 29.32 |
| I2VGenXL [47] | 2846.4 | 0.4 | 27.28 |
| DynamiCrafterXL [42] | 176.7 | 1.3 | 27.79 |
| SEINE [6] | 718.9 | 0.28 | 30.13 |
| SVD [3] | 857.2 | 1.1 | 23.95 |
| FreeNoise [25] | 1298.4 | **0** | 31.55 |
| OpenSora [48] | 1165.7 | 0.16 | 31.54 |
| OpenSoraPlan [24] | 72.9 | 0.24 | 29.34 |
| **StreamingT2V (Ours)** | **52.3** | 0.04 | **31.73** |

**Analysis of Table 1:**
*   **MAWE:** StreamingT2V (52.3) is vastly superior to SparseCtrl (6069.7) and I2VGenXL (2846.4), proving it combines consistency with actual motion. OpenSoraPlan is the closest competitor but still ~40% worse.
*   **SCuts:** StreamingT2V is nearly perfect (0.04). SparseCtrl's high score (5.48) confirms the hypothesis that "zero-padding" inputs creates artifacts.

    The following image (Figure 5 from the original paper) qualitatively shows the stagnation issue in baselines compared to the dynamic motion of StreamingT2V:

    ![该图像是示意图，左侧展示了一只在南极的松鼠在一堆榛子上的多个生成实例，右侧则呈现了一只在街道上吃生肉的老虎的不同生成效果。各个实例展示了不同方法对图像生成质量的影响。](images/5.jpg)
    *该图像是示意图，左侧展示了一只在南极的松鼠在一堆榛子上的多个生成实例，右侧则呈现了一只在街道上吃生肉的老虎的不同生成效果。各个实例展示了不同方法对图像生成质量的影响。*

## 6.3. Ablation Studies
The authors verified the contribution of each component:

1.  **CAM vs. Concatenation:**
    *   Baseline "Add-Cond" (ControlNet style with zero-convolution but masking): SCuts = 0.284.
    *   Baseline "Conc-Cond" (Direct channel concatenation): SCuts = 0.24.
    *   **Ours (CAM):** SCuts = 0.03.
    *   *Conclusion:* Simply concatenating frames or using standard ControlNet methods is insufficient for seamless transitions. The **attention mechanism** in CAM is crucial.

        The following figure (Figure 16 from the original paper) illustrates the "Add-Cond" baseline used for this comparison:

        ![该图像是示意图，展示了三种视频生成方法的对比：上行展示了使用 'Naive Concatenation'（a）、'Shared Noise'（b）和 'Randomized Blending'（c）方法生成视频的效果，比较不同方法在 XT 切片上的平滑性与一致性。](images/16.jpg)
        *该图像是示意图，展示了三种视频生成方法的对比：上行展示了使用 'Naive Concatenation'（a）、'Shared Noise'（b）和 'Randomized Blending'（c）方法生成视频的效果，比较不同方法在 XT 切片上的平滑性与一致性。*

2.  **APM (Long-Term Memory):**
    *   Without APM: Person Re-ID score = 93.42. LPIPS (Image Distance) = 0.192.
    *   **With APM:** Person Re-ID score = 94.95. LPIPS = 0.151.
    *   *Conclusion:* APM improves identity preservation by ~1.5% and scene consistency (LPIPS) by over 20%.

3.  **Randomized Blending:**
    *   Baseline B (Independent chunks): Temporal smoothness score (std dev of flow) = 8.72 (High flicker).
    *   Baseline B+S (Shared Noise): Score = 6.01.
    *   **Ours (Randomized Blending):** Score = 3.32.
    *   *Conclusion:* Randomized blending reduces transition flicker by 62% compared to naive enhancement.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
StreamingT2V represents a significant step forward in long-form text-to-video generation. By identifying the weaknesses of naive autoregression (stagnation and forgetting), the authors introduced a dual-memory system: **CAM** for seamless short-term motion and **APM** for long-term identity anchoring. Combined with **Randomized Blending** for high-quality upscaling, the system can generate videos of theoretically infinite length that remain dynamic and consistent. The proposed metric, **MAWE**, also provides a more nuanced way to evaluate video generation by penalizing the "easy way out" of generating static videos.

## 7.2. Limitations & Future Work
*   **Base Model Dependency:** The current implementation relies on ModelScope (UNet-based). While the paper shows a proof-of-concept with OpenSora (DiT-based), the primary results are tied to the quality of the underlying short-video generator.
*   **Inference Speed:** The autoregressive nature means generation time scales linearly with video length. The separate refinement stage adds further computational cost.
*   **Future Work:** The authors explicitly mention extending the framework to **Diffusion Transformers (DiT)** like OpenSora, suggesting that the CAM/APM principles are architecture-agnostic.

## 7.3. Personal Insights & Critique
*   **The "Stagnation" Insight is Key:** The paper identifies a subtle but critical failure mode in AI video—that models often default to "doing nothing" to minimize error. The MAWE metric is a valuable contribution for the community to ensure we aren't just praising high-quality static images masquerading as video.
*   **Elegant Memory Design:** The separation of memory into "motion" (CAM/Attention) and "appearance" (APM/Anchor) is intuitive and effective. It mimics how humans process video: we track movement frame-to-frame but remember the actor's identity from the start.
*   **Simplicity of Randomized Blending:** The blending technique is a clever, low-tech solution to a hard problem. Instead of training a complex new model to stitch videos, they exploited the stochastic nature of diffusion. This is a highly transferable technique applicable to any tiled diffusion task (e.g., high-res image generation).