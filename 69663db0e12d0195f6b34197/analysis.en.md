# 1. Bibliographic Information

## 1.1. Title
Self-Forcing++: Towards Minute-Scale High-Quality Video Generation

The title clearly indicates the paper's central focus: a method named `Self-Forcing++` designed to generate long-duration ("minute-scale"), high-quality videos. It positions itself as an advancement over a previous method, likely "Self-Forcing".

## 1.2. Authors
Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh.

The affiliations listed are UCLA, ByteDance Seed, and the University of Central Florida. The presence of researchers from both a top academic institution (UCLA) and a leading technology company (ByteDance) suggests a collaboration that combines rigorous academic research with industry-scale resources and objectives. Cho-Jui Hsieh (corresponding author) is a notable professor at UCLA specializing in machine learning optimization and large-scale models.

## 1.3. Journal/Conference
The paper is available as a preprint on arXiv. An arXiv submission is a common way for researchers to quickly disseminate their findings to the public before or during the formal peer-review process for a conference or journal. At the time of this analysis, the paper has not yet been published in a peer-reviewed venue.

## 1.4. Publication Year
The paper specifies a future publication date of October 2, 2025 (`2025-10-02T17:55:42.000Z`). This is likely a placeholder date used in the arXiv submission system. The content and references to other 2025 works indicate it is a very recent piece of research.

## 1.5. Abstract
The abstract introduces the core problem: diffusion models, while excellent for visual quality, are computationally expensive for generating long videos due to their transformer architectures. The paper focuses on autoregressive methods, which generate videos sequentially but suffer from quality degradation when extending beyond their training horizon. This degradation arises from the compounding of errors.

The proposed solution, `Self-Forcing++`, aims to solve this without needing long videos for training. The key idea is to use a powerful short-horizon "teacher" model to provide guidance to a "student" model on segments sampled from the student's *own self-generated long videos*. This process teaches the student to correct its own accumulated errors. The method is shown to scale video length by up to 20-50 times beyond the baseline, achieving generation of up to 4 minutes and 15 seconds. The authors also propose a new benchmark to address flaws in existing ones and demonstrate that their approach significantly outperforms baselines in both quality (`fidelity`) and temporal `consistency`.

## 1.6. Original Source Link
*   **Original Source:** [https://arxiv.org/abs/2510.02283](https://arxiv.org/abs/2510.02283)
*   **PDF Link:** [https://arxiv.org/pdf/2510.02283v1](https://arxiv.org/pdf/2510.02283v1)
*   **Publication Status:** Preprint on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The primary challenge addressed by this paper is the **generation of long, temporally coherent, and high-quality videos**. While diffusion models like Sora have achieved stunning results, they are typically limited to short clips (5-10 seconds). This is because their underlying transformer architectures process all frames simultaneously (bidirectionally), and the computational cost grows quadratically with the number of frames, making long video generation prohibitively expensive.

A promising alternative is the **autoregressive approach**, where video is generated frame-by-frame or chunk-by-chunk, conditioning each new part on the previous ones. This is much more scalable. However, autoregressive models face a critical problem: **error accumulation**. Minor prediction errors in early frames compound over time, leading to a catastrophic decline in video quality. This often manifests as visual artifacts like over-exposure, darkening, motion freezing, or a complete collapse into noise.

The paper identifies a crucial gap in existing research, which it terms a **training-inference misalignment**:
1.  **Temporal Mismatch:** Models are trained on short video clips (e.g., 5 seconds), the maximum length a powerful "teacher" model can handle, but are expected to generate much longer videos during inference.
2.  **Supervision Mismatch:** During training, the student model receives perfect guidance from the teacher for every frame. It is never exposed to the kind of degraded, error-ridden inputs it will inevitably generate and have to use as context during a long inference rollout.

    The paper's innovative entry point is to directly address this mismatch. Instead of avoiding errors, the authors propose a method that **forces the student model to learn how to recover from them**. The core idea is to let the student generate long, imperfect videos and then use the short-horizon teacher to correct sampled segments of these degraded rollouts, effectively teaching the student self-correction.

## 2.2. Main Contributions / Findings
The paper makes several key contributions:

1.  **Identifies the Core Bottleneck:** It clearly articulates that the dual temporal and supervision mismatch is the primary reason why autoregressive video models fail at long-horizon generation. This provides a clear problem definition for the field.
2.  **Proposes Self-Forcing++:** A simple and effective training framework to overcome this bottleneck. It extends video generation far beyond the teacher's limit by training the student on its own long, error-accumulated trajectories. This is achieved without needing any long videos for supervision or re-training on new datasets.
3.  **Achieves State-of-the-Art Performance and Scalability:** `Self-Forcing++` demonstrates a remarkable ability to scale. It generates high-quality videos up to 100 seconds, a 20x improvement over its baseline. Furthermore, with increased training computation, it can generate videos lasting over 4 minutes, showcasing unprecedented horizon scalability for this class of models.
4.  **Introduces a New Evaluation Metric:** The paper identifies a bias in the widely used `VBench` benchmark, which tends to favor over-exposed or static videos, leading to misleading scores. They propose a new metric, **`Visual Stability`**, which uses the advanced multimodal model `Gemini-2.5-Pro` to provide more reliable evaluations of long-video quality, specifically targeting error accumulation and exposure issues.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that create data (like images or videos) by reversing a noise-adding process. The core idea involves two steps:
*   **Forward Process:** Start with a clean data sample (e.g., a video frame) and gradually add a small amount of Gaussian noise over many timesteps, until the data becomes pure, indistinguishable noise. This process is fixed and does not involve learning.
*   **Reverse Process:** Train a neural network to reverse this process. At each timestep, the network takes a noisy input and predicts the noise that was added. By subtracting this predicted noise, it can gradually denoise the input, step-by-step, until a clean data sample is generated from an initial pure noise input. This learned denoising network is the heart of the model.

### 3.1.2. Diffusion Transformers (DiT)
Traditionally, the denoising network in diffusion models used a U-Net architecture. The **Diffusion Transformer (DiT)**, proposed by Peebles and Xie (2023), replaced the U-Net with a **Transformer** architecture. Transformers are known for their ability to capture long-range dependencies in data, which is crucial for understanding the global context of an image or the spatio-temporal relationships in a video. This architectural shift proved to be highly scalable and led to significant improvements in generation quality, forming the backbone of models like Sora.

### 3.1.3. Autoregressive Generation
Autoregressive modeling is a sequential generation process. An autoregressive model generates a sequence of data one element at a time, where the generation of each new element is conditioned on all the previously generated elements. For video, this means generating the next frame (or chunk of frames) based on the frames that have already been created. This is in contrast to bidirectional models (like a standard DiT) that process the entire sequence at once. Autoregressive models are computationally more efficient for long sequences but are susceptible to error accumulation.

### 3.1.4. KV Caching
In transformer-based autoregressive models, **KV Caching** is a critical optimization technique. During generation, a transformer calculates attention using Query (Q), Key (K), and Value (V) matrices. For each new token (or video frame) being generated, the K and V matrices for all *previous* tokens are needed. Instead of recomputing these matrices for the entire history at every step, KV caching stores them in memory. This way, the model only needs to compute the K and V for the newest token and append them to the cache, making the generation process much faster and enabling real-time streaming.

### 3.1.5. Knowledge Distillation
Knowledge distillation is a training paradigm where a smaller, more efficient "student" model is trained to replicate the output of a larger, more powerful "teacher" model. In the context of this paper, a multi-step, computationally expensive diffusion model (the teacher) is "distilled" into a student model that can generate high-quality results in just a few steps. This makes the generation process significantly faster.

## 3.2. Previous Works

### 3.2.1. CausVid
`CausVid` was a key step towards autoregressive video diffusion. It proposed a method to distill a powerful bidirectional teacher model into a streaming, autoregressive student model. The student uses a **KV cache** to maintain temporal context from previous frames. However, `CausVid` had two main drawbacks: it relied on recomputing overlapping frames between chunks to maintain consistency, and it suffered from a significant train-inference mismatch that often led to **over-exposure artifacts** in the generated videos.

### 3.2.2. Self-Forcing
`Self-Forcing` improved upon `CausVid` by better aligning the training and inference processes. It incorporated the KV cache directly into the training loop, training the model on its own generated rollouts (hence "self-forcing"). This effectively mitigated the over-exposure problem and set a new standard for quality in short-form autoregressive video generation. However, its crucial limitation, which `Self-Forcing++` addresses, is that it was only trained on short rollouts (e.g., 5 seconds), the maximum horizon of its teacher. When asked to generate beyond this limit, its quality would rapidly degrade due to unhandled error accumulation.

### 3.2.3. Distribution Matching Distillation (DMD)
DMD is the specific distillation technique used in this line of work. Instead of simply matching the final output (e.g., using an L2 loss), DMD aims to match the entire **distribution** of the student's generation trajectory with that of the teacher's at various noise levels. This is typically framed as minimizing the Kullback-Leibler (KL) divergence between the two distributions. The paper provides the formula for the DMD loss gradient in its appendix, which is crucial for understanding the optimization process (see Section 4.2.3).

### 3.2.4. Group Relative Policy Optimization (GRPO)
GRPO is a reinforcement learning (RL) algorithm adapted for fine-tuning generative models. In this paper, it is used as an optional final step to improve the long-term smoothness of the generated videos. It treats the generator as a "policy" and uses a "reward" signal to guide its updates. Outputs are generated in groups, and their rewards are compared to compute an advantage, which then informs the policy update. This helps the model learn preferences, such as favoring videos with smoother motion.

## 3.3. Technological Evolution
The field of video generation has evolved rapidly:
1.  **Early Models:** Initial video diffusion models were based on U-Net architectures, extending image generation techniques to the temporal domain. They were limited to short, often low-resolution clips.
2.  **Transformer Revolution:** The introduction of `DiT` led to models like Sora, Hunyuan-DiT, and Veo, which dramatically increased video quality, realism, and coherence by leveraging the scalability and global context modeling of transformers. However, these models remained non-streaming and computationally intensive for long videos.
3.  **The Rise of Autoregressive Methods:** To tackle the length limitation, researchers explored autoregressive formulations. Methods like `CausVid` and `Self-Forcing` introduced distillation and KV caching to create efficient, streaming models. These models achieved high quality on short videos but failed on long ones due to error accumulation.
4.  **Self-Forcing++:** This paper represents the next step, directly targeting the error accumulation problem by proposing a novel training scheme that teaches the model to recover from its own long-term generation errors.

## 3.4. Differentiation Analysis
The core innovation of `Self-Forcing++` lies in its training data and process, which sets it apart from its predecessors:
*   **vs. CausVid:** `Self-Forcing++` eliminates the need for recomputing overlapping frames and avoids the over-exposure artifacts of `CausVid` by using a rolling KV cache during both training and inference and by adopting the self-forcing paradigm.
*   **vs. Self-Forcing:** The key difference is the **horizon of the training rollouts**. `Self-Forcing` trains on *short* rollouts (e.g., 5 seconds) that are always within the teacher's comfort zone. In contrast, `Self-Forcing++` trains on *long* rollouts (e.g., 100 seconds) that are far beyond the teacher's horizon. By sampling and correcting segments from these long, degraded videos, it explicitly trains the model to handle the compounding errors that `Self-Forcing` was never exposed to during its training. This is the fundamental reason for its superior long-horizon performance.

# 4. Methodology

## 4.1. Principles
The central idea of `Self-Forcing++` is to **leverage a powerful short-horizon teacher to supervise a student on tasks the teacher itself cannot perform**, namely, correcting long-duration videos. The authors observe that even when a student model's long video generation degrades, it often retains some structural coherence. The problem is not a complete breakdown but an accumulation of errors.

The methodology rests on the intuition that any short, high-quality video clip can be seen as a sample from the marginal distribution of a valid, longer video. Therefore, a teacher model trained on a massive corpus of short clips implicitly holds the knowledge to judge the quality and correctness of *any* short segment, even one extracted from a longer, synthetically generated video. `Self-Forcing++` operationalizes this by having the student generate long videos, intentionally inviting error accumulation, and then using the teacher to guide the student on how to correct these errors, one short window at a time.

The complete generative process, from initial distillation to the final `Self-Forcing++` training loop and optional RL fine-tuning, is detailed below and summarized in the paper's Algorithm 1.

The following figure from the paper illustrates the workflow of `Self-Forcing++` compared to baselines.

![Figure 2 Workflow between baselines and Self-Forcing $^ { + + }$ . Our method employ backward noise initialization, extended DMD and rolling KV Cache to effectively mitigates train-test discrepancies.](images/2.jpg)
*该图像是示意图，展示了 Self-Forcing++ 方法与基线方法之间的工作流程。通过引入反向噪声初始化、扩展 DMD 和滚动 KV Cache，旨在有效减小训练与推理阶段之间的差异。*

## 4.2. Core Methodology In-depth

### 4.2.1. Stage 1: Initialization and Conversion (Background)
Before the main `Self-Forcing++` training can begin, the base model needs to be prepared. This follows the procedure from prior works like `CausVid` and `Self-Forcing`.

1.  **Distillation:** A large, bidirectional, multi-step video diffusion model (the teacher) is distilled into a few-step student generator. This is done using **Distribution Matching Distillation (DMD)**, which minimizes the reverse KL divergence between the student and teacher distributions. The objective is to create a student that produces similar quality results in far fewer denoising steps.
2.  **Conversion to Autoregressive Model:** The distilled student model, which is still bidirectional, is converted into an autoregressive one by incorporating **causal attention** (i.e., a frame can only attend to past frames) and a **KV cache**. To initialize this autoregressive student, it is trained to replicate Ordinary Differential Equation (ODE) trajectories sampled from the teacher. This serves as a warm-up phase. The loss for this stage is given by:
    \$
    \mathcal { L } _ { \mathrm { ode } } = \mathbb { E } _ { \mathbf { x } , t } \left[ \left\| G _ { \phi } \left( \{ \mathbf { x } _ { t _ { i } } ^ { ( i ) } \} _ { i = 1 } ^ { N } , \{ t _ { i } \} _ { i = 1 } ^ { N } \right) - \{ \mathbf { x } _ { \mathrm { teacher } } ^ { ( i ) } \} _ { i = 1 } ^ { N } \right\| ^ { 2 } \right]
    \$
    *   $G_\phi$: The autoregressive student model with parameters $\phi$.
    *   $\{ \mathbf { x } _ { t _ { i } } ^ { ( i ) } \}_{i=1}^N$: A sequence of noisy latents at different timesteps $\{t_i\}_{i=1}^N$.
    *   $\{ \mathbf { x } _ { \mathrm { teacher } } ^ { ( i ) } \}_{i=1}^N$: The corresponding denoised latents produced by the teacher model.
    *   This is a simple mean squared error loss that trains the student to match the teacher's output on short sequences.

### 4.2.2. Stage 2: The Self-Forcing++ Training Loop
This is the core contribution of the paper. The loop consists of three main steps: long rollout, backward noise initialization, and extended DMD.

**Step 1: Long Autoregressive Rollout**
The student generator $G_\theta$ is used to produce a long video sequence of $N$ clean frames, where $N$ is significantly larger than the teacher's capability $T$ (e.g., $N=100$s, $T=5$s). This is done autoregressively, using a **rolling KV cache** of a fixed size $L$. This generated video, denoted $\{x_t^S\}_{t=1}^N$, will likely contain accumulated errors, especially in later frames.

**Step 2: Backward Noise Initialization**
The key challenge is how to use the teacher to correct this long video. A segment is first sampled from the long rollout. Instead of starting the denoising process from pure random noise (which would lose all temporal context from the preceding frames), noise is added *back* to the clean frames of the sampled segment. This creates a noisy but contextually relevant starting point for the teacher and student. The process of adding noise back to a clean sample $x_0$ to get a noisy sample $x_t$ at timestep $t$ is generally described by:
\$
x _ { t } = ( 1 - \sigma _ { t } ) x _ { 0 } + \sigma _ { t } \epsilon , \quad \mathrm { where } \ \epsilon \sim \mathcal { N } ( 0 , I )
\$
Here, $x_0$ is a frame from the student's long rollout. The denoised estimate of $x_0$ itself is predicted from the previous step's latent $x_{t-1}$:
\$
x _ { 0 } = x _ { t - 1 } - \sigma _ { t - 1 } \hat { \epsilon } _ { \theta } ( x _ { t - 1 } , t - 1 )
\$
*   $\sigma_t$: The noise level at timestep $t$ from a predefined noise schedule.
*   $\epsilon$: Standard Gaussian noise.
*   $\hat{\epsilon}_\theta$: The noise prediction network (the student model).

    This ensures that the correction process is grounded in the temporal context of the long video.

**Step 3: Extended Distribution Matching Distillation (DMD)**
With the contextually relevant noisy segment, both the student $G_\theta$ and the teacher $T_\phi$ are used to denoise it. The `Self-Forcing++` loss then minimizes the distributional discrepancy between their outputs. This is done by sampling a contiguous window of length $K$ (the teacher's horizon) from the long student rollout of length $N$. The loss is the average KL divergence over all possible windows. The gradient of this loss is formulated as:
\$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { \mathrm { \scriptsize ~ { \mathrm { extended } } } } = \mathbb { E } _ { t } \mathbb { E } _ { z } \left[ \nabla _ { \theta } \mathrm { K L } \Big ( p _ { \theta , t } ^ { S } ( z ) \| p _ { t } ^ { T } ( z ) \Big ) \right] } \\ & { \qquad \approx - \mathbb { E } _ { t } \mathbb { E } _ { i \sim \mathrm { Unif } \ \{ 1 , \dots , N - K + 1 \} } \left[ \int \Bigl ( s ^ { T } ( \Phi ( G _ { \theta } ( z _ { i } ) , t ) , t ) - s _ { \theta } ^ { S } ( \Phi ( G _ { \theta } ( z _ { i } ) , t ) , t ) \Big ) \frac { d G _ { \theta } ( z _ { i } ) } { d \theta } d z _ { i } \right] , } \end{array}
\$
*   $G_\theta(z)$: The student generator's rollout.
*   $i \sim \mathrm{Unif}\{1, \dots, N-K+1\}$: A starting index for a window of length $K$, sampled uniformly from the long rollout of length $N$. This is the **key extension**—the model is trained on segments from anywhere in the long video.
*   $\Phi(\cdot, t)$: The noising process that takes a clean video segment and adds noise corresponding to timestep $t$.
*   $s^T$ and $s_\theta^S$: The score functions (related to the noise prediction) of the teacher and student models, respectively. The difference between these scores drives the learning.
*   By minimizing this loss, the student model $G_\theta$ is trained to produce outputs that are distributionally identical to the powerful teacher, even for segments deep into a long, self-generated video. This teaches it to recover from degraded states.

**Step 4: Training with a Rolling KV Cache**
A crucial detail is the use of a **rolling KV cache** during both training (in the long rollout step) and inference. This eliminates the train-inference mismatch that was a partial issue in `Self-Forcing`. It simplifies the entire pipeline, removing the need for recomputing overlapping frames (`CausVid`) or latent frame masking (`Self-Forcing`).

### 4.2.3. Stage 3: Improving Long-Term Smoothness via GRPO (Optional)
To further enhance temporal consistency and prevent abrupt scene changes, the trained model can be fine-tuned using `Group Relative Policy Optimization (GRPO)`.
*   **Reward Function:** The reward signal is derived from the **optical flow** between consecutive frames. Optical flow measures the motion of objects between frames. A large, sudden change in optical flow magnitude indicates a jarring transition, which is penalized. Smoother transitions receive a higher reward.
*   **GRPO Update:** The model is updated to maximize the expected reward. The GRPO objective function is:
    \$
    \mathcal { I } ( \theta ) = \mathbb { E } _ { \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { old } } } ( \cdot | c ) } \mathbb { E } _ { a _ { t , i } \sim \pi _ { \theta _ { \mathrm { old } } } ( \cdot | s _ { t , i } ) } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { min } \Bigl ( \rho _ { t , i } A _ { i } , \mathrm { clip } ( \rho _ { t , i } , 1 - \epsilon , 1 + \epsilon ) A _ { i } \Bigr ) \right]
    \$
    *   $\pi_{\theta_{old}}$: The policy (generator) before the update.
    *   $\rho_{t,i} = \frac{\pi_\theta(a_{t,i} | s_{t,i})}{\pi_{\theta_{old}}(a_{t,i} | s_{t,i})}$: The importance sampling ratio between the new and old policies.
    *   $A_i$: The **advantage** for output $i$, calculated by normalizing its reward relative to the average reward of a group of $G$ generated outputs.
    *   `clip(...)`: A clipping function (from PPO algorithm) to prevent excessively large policy updates and stabilize training.
        This RL fine-tuning step encourages the model to generate videos that are not only high-quality frame-by-frame but also smooth and consistent over long durations.

# 5. Experimental Setup

## 5.1. Datasets
*   **Training Dataset:** The model was trained on a modified version of the **VidProM** dataset. VidProM is a large-scale dataset containing millions of real prompt-gallery pairs for text-to-video models. The authors used a version that was filtered and extended with a Large Language Model (LLM), similar to the setup in the `Self-Forcing` paper. Importantly, the `Self-Forcing++` training phase itself does not require real video data, as it trains on self-generated rollouts; this dataset is for the initial model pre-training/distillation.
*   **Evaluation Datasets (Prompts):**
    *   For short video (5s) evaluation, they used **946 prompts** from the `VBench` benchmark.
    *   For long video (50s, 75s, 100s) evaluation, they used **128 prompts** from the `MovieGen` dataset.

## 5.2. Evaluation Metrics
The paper uses several metrics to evaluate video quality, consistency, and text alignment.

### 5.2.1. VBench Metrics
`VBench` is a comprehensive suite of metrics. The paper reports several key ones:
*   **Text Alignment:** Measures how well the generated video matches the input text prompt. This is often calculated using `CLIP Score`.
    *   **Conceptual Definition:** It computes the cosine similarity between the embeddings of the video frames and the text prompt, produced by a pretrained multimodal model like CLIP. A higher score means better alignment.
    *   **Mathematical Formula:** $ \text{CLIPScore}(V, P) = w \cdot \mathbb{E}_{f \in V} [\cos(\text{Emb}_I(f), \text{Emb}_T(P))] $, where $w$ is a scaling factor, $V$ is the set of frames, $P$ is the prompt, and $\text{Emb}_I, \text{Emb}_T$ are the image and text encoders.
*   **Temporal Quality:** An aggregate score measuring the temporal consistency of the video, including subject and background consistency, and motion smoothness.
*   **Dynamic Degree:**
    *   **Conceptual Definition:** Measures the amount of motion in the video. A very low score indicates that the video has frozen or become static, a common failure mode in long video generation. A high score indicates sustained motion.
    *   **Mathematical Formula:** It is typically calculated based on the magnitude of the optical flow vectors between consecutive frames. $ \text{DynamicDegree} = \frac{1}{N-1} \sum_{i=1}^{N-1} ||\text{Flow}(f_i, f_{i+1})||_2 $.
    *   **Symbol Explanation:** $N$ is the number of frames, $f_i$ is the $i$-th frame, and $\text{Flow}(\cdot, \cdot)$ is the optical flow estimation function.
*   **Framewise Quality:** Assesses the visual quality of individual frames, independent of temporal aspects. This often includes measures of aesthetic quality and imaging quality (e.g., clarity, lack of artifacts).

### 5.2.2. Visual Stability (Proposed Metric)
*   **Conceptual Definition:** This novel metric is designed to overcome the limitations of `VBench`, which the authors found to be biased towards over-exposed and degraded long videos. `Visual Stability` uses a state-of-the-art Video MLLM (`Gemini-2.5-Pro`) to assess long-term quality degradation. The MLLM is prompted to rate a video on a 0-100 scale based on a detailed rubric that defines specific failure modes like over-exposure, under-exposure, and catastrophic quality collapse.
*   **Methodology:** The MLLM is given a detailed prompt with a 6-point scale (0: Catastrophic to 5: Well-Exposed) and asked to provide reasoning for its score. This qualitative and quantitative feedback provides a more reliable assessment of how well a model maintains visual quality over long durations. The results are then aggregated to a 0-100 scale. The paper shows that this metric aligns well with human judgment.

    The figure below shows how existing `VBench` metrics can give misleadingly high scores to degraded or over-exposed frames, motivating the need for `Visual Stability`.

    ![该图像是一个示意图，展示图像质量在不同曝光条件下的变化。左侧为严重退化的样本，质量评分分别为77.13和69.03；中间是轻微退化的图像，评分为67.29和53.64；右侧为正常曝光和过度曝光的图像，评分分别为43.03、50.88、59.05和64.01。](images/3.jpg)
    *该图像是一个示意图，展示图像质量在不同曝光条件下的变化。左侧为严重退化的样本，质量评分分别为77.13和69.03；中间是轻微退化的图像，评分为67.29和53.64；右侧为正常曝光和过度曝光的图像，评分分别为43.03、50.88、59.05和64.01。*

## 5.3. Baselines
The proposed method is compared against a comprehensive set of baselines:
*   **Autoregressive Models:**
    *   `NOVA`: An autoregressive model that models video synthesis without vector quantization.
    *   `Pyramid Flow`: An autoregressive model using a hierarchical flow matching process.
    *   `MAGI-1`: A large-scale autoregressive video generation model.
    *   `SkyReels-V2`: A model that uses the `diffusion forcing` technique to enable long rollouts.
    *   `CausVid`: The direct predecessor that introduced streaming generation via distillation.
    *   `Self-Forcing`: The immediate baseline that improved upon `CausVid` but was limited to short-horizon generation.
*   **Bidirectional Models (for reference):**
    *   `LTX-Video` and `Wan2.1`: State-of-the-art non-autoregressive models included to provide a quality ceiling for short (5s) videos. `Wan2.1` is also the base model used for distillation in `Self-Forcing++`.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results strongly validate the effectiveness of `Self-Forcing++`, particularly in its primary goal of long-horizon video generation.

### 6.1.1. Performance on Short (5s) and Long (50s) Videos
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">#Params</th>
<th rowspan="2">Throughput (S)</th>
<th colspan="3">Results on 5s ↑</th>
<th colspan="5">Results on 50s ↑</th>
</tr>
<tr>
<th>Total Score</th>
<th>Quality Score</th>
<th>Semantic Score</th>
<th>Text Alignment</th>
<th>Temporal Quality</th>
<th>Dynamic Degree</th>
<th>Visual Stability</th>
<th>Framewise† Quality</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10"><strong>Bidirectional models</strong></td>
</tr>
<tr>
<td>LTX-Video</td>
<td>1.9B</td>
<td>8.98</td>
<td>80.00</td>
<td>82.30</td>
<td>70.79</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Wan2.1</td>
<td>1.3B</td>
<td>0.78</td>
<td>84.67</td>
<td>85.69</td>
<td>80.60</td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td>-</td>
</tr>
<tr>
<td colspan="10"><strong>Autoregressive models</strong></td>
</tr>
<tr>
<td>NOVA</td>
<td></td>
<td>0.88</td>
<td>80.12</td>
<td>80.39</td>
<td>79.05</td>
<td>24.58</td>
<td>86.53</td>
<td>31.96</td>
<td>45.94</td>
<td>34.45</td>
</tr>
<tr>
<td>Pyramid Flow</td>
<td>0.6B 2B</td>
<td>6.7</td>
<td>81.72</td>
<td>84.74</td>
<td>69.62</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>MAGI-1</td>
<td>4.5B</td>
<td>0.19</td>
<td>79.18</td>
<td>82.04</td>
<td>67.74</td>
<td>26.04</td>
<td>88.34</td>
<td>28.49</td>
<td>51.25</td>
<td>54.20</td>
</tr>
<tr>
<td>SkyReels-V2</td>
<td>1.3B</td>
<td>0.49</td>
<td>82.67</td>
<td>84.70</td>
<td>74.53</td>
<td>23.73</td>
<td>88.78</td>
<td>39.15</td>
<td>60.41</td>
<td>54.13</td>
</tr>
<tr>
<td>CausVid</td>
<td>1.3B</td>
<td>17.0</td>
<td>82.46</td>
<td>83.61</td>
<td>77.84</td>
<td>25.25</td>
<td>89.34</td>
<td>37.35</td>
<td>40.47</td>
<td>61.56</td>
</tr>
<tr>
<td>Self Forcing</td>
<td>1.3B</td>
<td>17.0</td>
<td>83.00</td>
<td>83.71</td>
<td>80.14</td>
<td>24.77</td>
<td>88.17</td>
<td>34.35</td>
<td>40.12</td>
<td>61.06</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td><strong>1.3B</strong></td>
<td><strong>17.0</strong></td>
<td><strong>83.11</strong></td>
<td><strong>83.79</strong></td>
<td><strong>80.37</strong></td>
<td><strong>26.37</strong></td>
<td><strong>91.03</strong></td>
<td><strong>55.36</strong></td>
<td><strong>90.94</strong></td>
<td><strong>60.82</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Short-Horizon (5s):** `Self-Forcing++` performs comparably to the best baseline, `Self-Forcing`, and the strong bidirectional model `Wan2.1`. This confirms that the new training strategy does not compromise its ability to generate high-quality short videos.
*   **Long-Horizon (50s):** The superiority of `Self-Forcing++` is stark. Its **`Visual Stability` score is 90.94**, more than double that of `CausVid` (40.47) and `Self-Forcing` (40.12). This quantitatively confirms its ability to avoid the quality degradation (over-exposure, darkening) that plagues the baselines. Its **`Dynamic Degree` is 55.36**, significantly higher than all others, indicating it successfully maintains motion and avoids freezing.

### 6.1.2. Performance on Even Longer Videos (75s and 100s)
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="5">Results on 75s ↑</th>
<th colspan="5">Results on 100s ↑</th>
</tr>
<tr>
<th>Text Alignment</th>
<th>Temporal Quality</th>
<th>Dynamic Degree</th>
<th>Visual Stability</th>
<th>Framewise Quality</th>
<th>Text Alignment</th>
<th>Temporal Quality</th>
<th>Dynamic Degree</th>
<th>Visual Stability</th>
<th>Framewise Quality</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10"><strong>Autoregressive models</strong></td>
</tr>
<tr>
<td>NOVA</td>
<td>23.37</td>
<td>86.32</td>
<td>31.24</td>
<td>34.06</td>
<td>31.53</td>
<td>22.89</td>
<td>86.24</td>
<td>31.09</td>
<td>32.97</td>
<td>31.03</td>
</tr>
<tr>
<td>MAGI-1</td>
<td>24.95</td>
<td>87.89</td>
<td>24.82</td>
<td>43.28</td>
<td>52.04</td>
<td>23.75</td>
<td>87.62</td>
<td>22.21</td>
<td>39.38</td>
<td>50.90</td>
</tr>
<tr>
<td>SkyReels-V2</td>
<td>22.70</td>
<td>88.99</td>
<td>39.89</td>
<td>55.47</td>
<td>51.55</td>
<td>22.05</td>
<td>88.80</td>
<td>38.75</td>
<td>56.72</td>
<td>50.48</td>
</tr>
<tr>
<td>CausVid</td>
<td>24.76</td>
<td>89.14</td>
<td>35.82</td>
<td>39.84</td>
<td>60.96</td>
<td>24.41</td>
<td>89.06</td>
<td>34.60</td>
<td>39.21</td>
<td>61.01</td>
</tr>
<tr>
<td>Self Forcing</td>
<td>23.39</td>
<td>87.79</td>
<td>29.15</td>
<td>35.00</td>
<td>60.02</td>
<td>22.00</td>
<td>87.39</td>
<td>26.41</td>
<td>32.03</td>
<td>58.25</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td><strong>26.31</strong></td>
<td><strong>91.00</strong></td>
<td><strong>55.62</strong></td>
<td><strong>86.10</strong></td>
<td><strong>60.67</strong></td>
<td><strong>26.04</strong></td>
<td><strong>90.87</strong></td>
<td><strong>54.12</strong></td>
<td><strong>84.22</strong></td>
<td><strong>60.66</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
The performance gap widens as the video length increases. At 100 seconds, the `Dynamic Degree` of `Self-Forcing` drops to 26.41, confirming its tendency to freeze. In contrast, `Self-Forcing++` maintains a high `Dynamic Degree` of 54.12 and a `Visual Stability` of 84.22, demonstrating robust and sustained quality.

The figure below provides a qualitative comparison, visually showing how baseline methods like `CausVid` and `Self-Forcing` degrade over 100 seconds, while `Self-Forcing++` maintains high fidelity.

![该图像是展示不同视频生成模型在不同时间点 (t=0s, t=25s, t=50s, t=75s, t=100s) 生成的水下场景的比较图。该图包含SkyReels、MAGI-1、CausVid、Self Forcing以及我们的模型生成的图像，显示了在长视频生成中的质量与一致性差异。](images/4.jpg)
*该图像是展示不同视频生成模型在不同时间点 (t=0s, t=25s, t=50s, t=75s, t=100s) 生成的水下场景的比较图。该图包含SkyReels、MAGI-1、CausVid、Self Forcing以及我们的模型生成的图像，显示了在长视频生成中的质量与一致性差异。*

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Effect of Attention Window Length
The authors investigated whether simply shortening the attention window in the original `Self-Forcing` could mitigate error accumulation by forcing the model to rely on more varied cache states.

The following are the results from Table 3 of the original paper:

| Causvid | Self-Forcing | Attn-15 | Attn-12 | Attn-9 | Ours |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 40.47 | 40.12 | 44.69 | 42.19 | 52.50 | **90.94** |

**Analysis:**
Reducing the attention window size (from the default of 21 latent frames) provides a modest improvement in `Visual Stability` (from 40.12 to 52.50 for `Attn-9`). However, this is still far below the performance of `Self-Forcing++` (90.94) and comes at the cost of reduced consistency, as the model has less context to draw from. This shows that this simple fix is insufficient and the explicit long-horizon training of `Self-Forcing++` is necessary.

### 6.2.2. Effect of GRPO
This study demonstrates the benefit of the optional RL fine-tuning step.

![Figure 5 Comparison of generation outcomes with and without GRPO. Variance is computed with window size 8.](images/5.jpg)

**Analysis:**
The figure plots the magnitude of optical flow over time. Without GRPO (red line), the plot shows sharp spikes, which correspond to abrupt, unnatural scene transitions in the video. After fine-tuning with GRPO using an optical-flow-based reward (blue line), these spikes are suppressed. This leads to smoother motion and improved long-range temporal consistency, enhancing the overall perceptual quality of the videos.

### 6.2.3. Training Budget Scaling
This experiment is crucial as it explores the scalability of the proposed method. The authors trained the model with increasing amounts of computation (training budget).

![FigureScaling phenomenon observe in 55-second generation or prompt:A massive elephant walks sowlc a sunlit savannah, dust rising around its feet, the warm glow of sunset...".](images/6.jpg)

**Analysis:**
The results show a clear scaling phenomenon:
*   **Baseline (1x budget):** The model can generate a coherent 5-second video but quickly degrades on longer sequences, similar to `Self-Forcing`.
*   **4x budget:** The model maintains semantic coherence for longer (e.g., the elephant remains an elephant) but still suffers from quality degradation.
*   **8x budget:** The model begins to generate more detailed and semantically accurate content, but motion is still limited.
*   **20x budget:** A significant leap in quality, producing high-fidelity videos stable for over 50 seconds.
*   **25x budget:** The model achieves generation of a 255-second (over 4 minutes) video with negligible quality loss.

    This demonstrates that the `Self-Forcing++` framework is highly scalable. With more training, it progressively learns to handle longer and longer horizons, suggesting a viable path towards very long-duration video synthesis without requiring new datasets. The figure below highlights the impressive final capability of the model.

    ![Figure 1 Self-forcing $^ { + + }$ generates videos up to four minutes long. The radar chart highlights our model's superiority, while the line plot shows its sustained motion dynamics over long durations.](images/1.jpg)
    *该图像是图表，展示了自强 $^{++}$ 模型生成长达四分钟的视频能力。上方的四个画面分别显示了视频的四个时间段（0-60秒、60-120秒、120-180秒、180-240秒），下方的雷达图突出了该模型在多个指标上的优越性，尤其在时间质量与整体一致性方面。图表右下角的曲线图展示了不同模型在动态度量上的表现随时间的变化。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper introduces `Self-Forcing++`, a simple yet powerful method for training autoregressive video diffusion models to generate high-quality, minute-scale videos. By identifying the critical train-inference mismatch as the root cause of error accumulation, the authors propose a novel training scheme where a student model learns to correct errors by being supervised by a short-horizon teacher on its own long, self-generated rollouts.

This approach successfully extends the generation horizon by over 50x compared to its baseline, achieving state-of-the-art performance on long-video benchmarks. The paper also contributes a more reliable evaluation metric, `Visual Stability`, to address biases in existing benchmarks. The work paves the way for more robust and scalable long-video synthesis.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations and outline future research directions:
*   **Limitations:**
    *   **Training Speed:** The self-rollout process during training is computationally more expensive than standard teacher-forcing methods.
    *   **Lack of Long-Term Memory:** The model relies on a fixed-size rolling KV cache, which means it can "forget" content that is occluded or off-screen for an extended period, leading to potential inconsistencies.
    *   **Inherited Base Model Capacity:** The final quality is still capped by the capabilities of the underlying base model (`Wan2.1-1.3B`).
*   **Future Work:**
    *   **Training Efficiency:** Explore parallelizing the training process to reduce costs.
    *   **Fidelity Control:** Investigate techniques like quantizing or normalizing the KV cache to prevent distributional shift and further mitigate quality degradation.
    *   **Long-Term Memory:** Incorporate explicit long-term memory mechanisms into the autoregressive framework to achieve true long-range coherence.

## 7.3. Personal Insights & Critique
`Self-Forcing++` presents a compelling and elegant solution to a significant problem in generative AI.
*   **Strengths:**
    *   **Conceptual Simplicity:** The core idea of "learning from your own long mistakes" is intuitive and powerful. It cleverly circumvents the need for expensive, hard-to-acquire long-video datasets.
    *   **Demonstrated Scalability:** The training budget scaling experiment is one of the most exciting results. It suggests that the limitation is not fundamental to the model architecture but can be overcome with more computation, providing a clear path for future improvements.
    *   **Contribution to Evaluation:** The critique of existing benchmarks and the proposal of `Visual Stability` is a valuable contribution in its own right. As generative models improve, robust evaluation becomes increasingly critical, and leveraging powerful MLLMs for this task is a promising direction.

*   **Potential Issues and Areas for Improvement:**
    *   **Computational Cost:** While effective, the training cost remains a significant barrier. Each training step requires a full autoregressive rollout, which is slow. More efficient approximations of this process could make the method more practical.
    *   **Reward Function for GRPO:** The reliance on optical flow as a reward for smoothness is a good starting point, but it is a low-level signal. It captures motion continuity but might not capture higher-level semantic or narrative consistency. More sophisticated reward models could lead to even better results.
    *   **True Long-Term Coherence:** As the authors note, the rolling KV cache is a fundamental limitation for true long-term memory. A story-like video where a character leaves and returns much later would be challenging for this model. Integrating hierarchical or compressed memory mechanisms will be the next major frontier for this line of research.

        Overall, `Self-Forcing++` is a significant step forward, shifting the focus from simply generating short, high-quality clips to building models that can maintain that quality over extended, minute-scale durations. It provides both a practical method and a clear research trajectory for the future of long-form video generation.