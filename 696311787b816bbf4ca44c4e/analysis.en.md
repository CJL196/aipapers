# 1. Bibliographic Information
## 1.1. Title
**Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation**

The title clearly outlines the paper's core contributions.
*   **`Efficient Streaming Video Generation`**: This identifies the primary application domain—generating video in a continuous, real-time fashion, crucial for interactive applications.
*   **`Reward Forcing`**: This is the name of the proposed framework, suggesting a mechanism that actively pushes the model towards desirable outcomes using a reward signal.
*   **`Rewarded Distribution Matching Distillation`**: This specifies the core technical innovation, a modification of a known technique (`Distribution Matching Distillation`) that incorporates rewards to guide the learning process.

## 1.2. Authors
The authors are Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, Yujun Shen, and Min Zhang.

Their affiliations include prestigious academic institutions (Zhejiang University, Shanghai Jiao Tong University - SJTU) and a major industrial research lab (Ant Group). This collaboration between academia and industry is common in cutting-edge AI research, as it combines theoretical rigor with the large-scale computational resources and practical focus of industry. Several authors have a strong publication record in generative models and computer vision.

## 1.3. Journal/Conference
The paper is available as a preprint on arXiv, with a listed publication date of December 4, 2025. This indicates it has likely been submitted for peer review at a top-tier computer science conference for the 2025 cycle, such as CVPR (Conference on Computer Vision and Pattern Recognition), NeurIPS (Conference on Neural Information Processing Systems), or ICLR (International Conference on Learning Representations). These venues are highly competitive and are considered the premier forums for publishing significant advances in machine learning and AI.

## 1.4. Publication Year
The preprint is dated 2025.

## 1.5. Abstract
The abstract summarizes that efficient streaming video generation is a critical but challenging task. Current methods distill large diffusion models into few-step generators with sliding window attention. To prevent error accumulation, they use initial frames as static "sink tokens," but this leads to a new problem: the generated video becomes overly dependent on the first frame, resulting in copied frames and a lack of motion.

To solve this, the paper introduces **`Reward Forcing`**, a framework with two main components:
1.  **`EMA-Sink`**: This mechanism replaces static sink tokens with dynamic ones. These tokens are initialized from the first frames but are continuously updated with an exponential moving average (EMA) of frames that are "evicted" from the sliding attention window. This allows the model to retain long-term context while incorporating recent dynamic information, preventing the "first-frame copying" issue.
2.  **`Re-DMD` (Rewarded Distribution Matching Distillation)**: This novel distillation technique improves upon standard distribution matching. Instead of treating all training samples equally, it uses a vision-language model (VLM) to score the "motion dynamics" of generated samples. It then uses these scores to prioritize samples with better motion during training, biasing the model's output towards more dynamic videos without sacrificing quality.

    The authors demonstrate through experiments that `Reward Forcing` achieves state-of-the-art performance, generating high-quality streaming video at **23.1 frames per second (FPS)** on a single NVIDIA H100 GPU.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2512.04678
*   **PDF Link:** https://arxiv.org/pdf/2512.04678v2.pdf
*   **Publication Status:** This is a preprint on arXiv and has not yet been officially published in a peer-reviewed venue.

# 2. Executive Summary
## 2.1. Background & Motivation
### 2.1.1. Core Problem
The central challenge addressed by this paper is the trade-off between **efficiency** and **quality** in long-form video generation. While large-scale video diffusion models can create stunning, high-fidelity short clips, their computational cost makes them unsuitable for "streaming" applications like interactive simulations or virtual worlds, where video must be generated continuously and with low latency.

### 2.1.2. Existing Gaps and Challenges
To achieve efficiency, recent research has focused on **autoregressive models**. These models generate video frame-by-frame (or chunk-by-chunk), attending only to past frames. This is much faster than "bidirectional" models that process all frames at once. However, this approach introduces significant challenges:

1.  **Error Accumulation:** In an autoregressive system, each new frame is generated based on previously generated frames. If a small error or artifact appears in one frame, it can be amplified as it propagates through subsequent frames, leading to a catastrophic decline in quality over time. This is often called "drifting."
2.  **The "Static Sink" Dilemma:** To combat error accumulation, methods like `LongLive` adopted an **attention sink** mechanism. They keep the initial frames of the video permanently in the attention window. This provides a stable, high-quality reference point that anchors the generation process and prevents drifting. However, this solution creates a new problem: **motion stagnation**. The model becomes overly reliant on these static initial frames, causing subsequent frames to look too similar to the beginning, effectively "copying the first frame" and killing any significant motion or scene evolution.
3.  **Motion-Agnostic Training:** The standard training method for these fast models is **distribution matching distillation (`DMD`)**, where a small "student" model learns to imitate the output of a powerful "teacher" model. However, `DMD` is blind to the semantic quality of the output. It will happily teach the student to replicate a visually coherent but static, boring video, as long as it matches the teacher's distribution. There is no built-in incentive to prioritize generating videos with rich and meaningful motion.

### 2.1.3. The Paper's Innovative Idea
The authors of `Reward Forcing` propose a two-pronged attack on these challenges. Instead of accepting the trade-off between consistency and motion, they aim to achieve both simultaneously.
*   For the "static sink" problem, they ask: What if the sink wasn't static? They propose **`EMA-Sink`**, a dynamic context window that smoothly blends long-term history with recent information, providing a stable anchor without being rigidly tied to the initial frame.
*   For the "motion-agnostic training" problem, they ask: How can we make the model *care* about motion? They introduce **`Re-DMD`**, which integrates a "reward" signal directly into the distillation process. This reward, provided by a powerful VLM that can judge motion quality, "forces" the model to pay more attention to learning from and generating dynamic content.

## 2.2. Main Contributions / Findings
The paper presents the following primary contributions:

1.  **A Novel Framework, `Reward Forcing`**: This is a comprehensive solution for efficient and high-quality streaming video generation that explicitly addresses motion stagnation.
2.  **`EMA-Sink` Mechanism**: This is a new, computationally efficient state-packaging mechanism for long video generation. By using an exponential moving average to update a fixed-size set of "sink" tokens, it preserves both long-term consistency and recent dynamics, effectively solving the first-frame copying problem seen in previous methods.
3.  **`Re-DMD` (Rewarded Distribution Matching Distillation)**: This is a novel distillation objective that enhances motion quality. It re-weights the standard `DMD` loss based on a motion-quality reward from a VLM. This steers the model to prioritize learning from and generating more dynamic videos while maintaining high fidelity to the original data distribution.
4.  **State-of-the-Art Performance**: The paper demonstrates that `Reward Forcing` achieves top performance on standard video generation benchmarks for both short (5-second) and long (60-second) videos. Crucially, it achieves this while being extremely efficient, capable of real-time generation at **23.1 FPS**, significantly outperforming prior autoregressive methods in both speed and quality.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
### 3.1.1. Video Diffusion Models
A **diffusion model** is a type of generative model that learns to create data by reversing a noise process.
*   **Forward Process:** You start with a clean piece of data (like a video frame) and gradually add a small amount of Gaussian noise over many steps, until it becomes pure, unrecognizable noise. This process is fixed and doesn't involve learning.
*   **Reverse Process:** The model, typically a neural network, is trained to reverse this process. At each step, it takes a noisy frame and predicts the noise that was added. By subtracting this predicted noise, it can gradually denoise the data, starting from pure noise and ending with a clean, realistic frame.

    For video, diffusion models can be:
*   **Bidirectional:** These models, like `Sora`, process all frames of a video clip simultaneously. Each frame's denoising process can "see" all other frames (both past and future). This allows for excellent global consistency and complex dynamics but is computationally very expensive and slow, making it unsuitable for real-time streaming.
*   **Autoregressive:** These models generate video one piece at a time (e.g., a single frame or a small chunk of frames). When generating frame $i$, the model can only see frames `1` to `i-1`. This is much faster and allows for generating videos of arbitrary length, but it is prone to error accumulation.

### 3.1.2. Model Distillation
**Model distillation** is a technique to transfer knowledge from a large, powerful, but slow "teacher" model to a smaller, faster "student" model. The goal is for the student to achieve performance close to the teacher's but with much lower computational cost. In the context of this paper, the teacher is a slow, multi-step diffusion model, and the student is a fast, few-step (often single-step) autoregressive generator.

### 3.1.3. Distribution Matching Distillation (DMD)
`DMD` is a specific and powerful distillation technique for diffusion models. Instead of just matching the final outputs, `DMD` aims to make the student's entire output *distribution* match the teacher's. It achieves this by matching their **score functions**.

*   **Score Function:** In diffusion models, the score function, $s(\mathbf{x}_t, t) = \nabla_{x_t} \log p(\mathbf{x}_t)$, essentially tells you which direction to move a noisy sample $\mathbf{x}_t$ to make it more likely under the true data distribution. It is directly related to the noise prediction task.
*   **DMD Loss:** `DMD` trains the student generator $G_\theta$ by minimizing the difference between the teacher's score function $s_{real}$ (which is fixed) and a student score function $s_{fake}$ that is trained on the outputs of the student generator. The gradient of the DMD loss is given by Eq. 1 in the paper:
    \$
    \nabla_{\theta} \mathcal{L}_{\mathrm{DMD}} \approx - \mathbb{E}_{t} \Big( \int (s_{\mathrm{real}}(\Psi(G_{\theta}(\epsilon), t), t) - s_{\mathrm{fake}}(\Psi(G_{\theta}(\epsilon), t), t)) \frac{\mathrm{d}G_{\theta}(\epsilon)}{\mathrm{d}\theta} \mathrm{d}\epsilon \Big)
    \$
    Here, $G_{\theta}(\epsilon)$ is a sample generated by the student, $\Psi(\cdot, t)$ adds noise to it to get a noisy sample at timestep $t$, and the term $(s_{real} - s_{fake})$ is the "correction" signal that tells the generator how to adjust its output to better match the teacher's distribution.

### 3.1.4. Sliding Window Attention & KV Cache
In transformer models, the `Attention` mechanism has a computational cost that is quadratic in the sequence length ($O(L^2)$). For long videos, this is prohibitively expensive.
*   **Sliding Window Attention:** To solve this, autoregressive models use a sliding window. When generating the current frame, the model only attends to a small, fixed number ($w$) of the most recent previous frames. This keeps the computational cost constant, regardless of the video's length.
*   **KV Cache:** To make this efficient, the `Key` (K) and `Value` (V) vectors calculated for each frame are stored in a `KV cache`. When generating the next frame, these cached values can be reused without recomputation, dramatically speeding up inference.

### 3.1.5. Attention Sink
A 2023 paper ("Efficient Streaming Language Models with Attention Sinks") discovered a peculiar property of transformer attention: the very first tokens in a sequence (the "attention sink" tokens) seem to absorb a disproportionate amount of attention, and removing them during sliding window attention causes a catastrophic drop in performance. Keeping these initial tokens in the `KV cache` permanently, even as the window slides, stabilizes the model and preserves performance. This technique was adopted by video models like `LongLive` to prevent quality drift, but as `Reward Forcing` points out, it leads to motion stagnation.

## 3.2. Previous Works
*   **`CausVid` and `Self-Forcing`**: These papers were foundational for efficient autoregressive video generation. `CausVid` first proposed using `DMD` to distill a bidirectional teacher into a causal (autoregressive) student. `Self-Forcing` improved upon this by simulating the inference-time conditions during training. Specifically, instead of always conditioning on pristine ground-truth frames, it conditions on frames generated by the model itself, bridging the "train-test gap" and making the model more robust to its own errors. `Reward Forcing` builds directly on this framework.
*   **`LongLive`**: This work extended the `Self-Forcing` idea to generate much longer videos. It introduced two key ideas: `KV recaching` for efficiency and using **static initial frames as attention sinks** to maintain long-term consistency. `LongLive` is the most direct predecessor that `Reward Forcing` aims to improve upon, specifically targeting the motion stagnation caused by its static sink.
*   **RL for Video Generation**: Other research has explored using Reinforcement Learning (RL) to align generative models with human preferences or other non-differentiable metrics. Methods like `DPO` (Direct Preference Optimization) and `RLHF` (Reinforcement Learning from Human Feedback) have been adapted for video. These typically involve a separate fine-tuning stage after initial training.

## 3.3. Technological Evolution
The field has progressed as follows:
1.  **Bidirectional Models (`Imagen Video`, `Wan2.1`):** High quality, but slow and limited to short clips.
2.  **Early Autoregressive Models:** Capable of longer generation, but often with lower quality and consistency issues.
3.  **Distilled Autoregressive Models (`CausVid`, `Self-Forcing`):** Achieved high speed (few-step generation) by distilling large teachers using `DMD`. They established the modern framework for efficient video generation.
4.  **Long Video Distilled Models (`LongLive`):** Extended the distilled approach to minute-long videos by tackling error accumulation with techniques like static attention sinks. However, this introduced the motion stagnation problem.
5.  **`Reward Forcing` (This Paper):** Represents the next step, aiming to solve the motion stagnation problem of long video models by introducing dynamic context management (`EMA-Sink`) and motion-aware training (`Re-DMD`).

## 3.4. Differentiation Analysis
*   **vs. `LongLive`**: The key difference is the sink mechanism. `LongLive` uses **static** sink tokens (the first frames never change). `Reward Forcing` uses **dynamic** `EMA-Sink` tokens that are continuously updated, blending historical and recent context.
*   **vs. `Self-Forcing`/`CausVid`**: The key difference is the distillation loss. `Self-Forcing` uses vanilla `DMD`, which is unaware of motion quality. `Reward Forcing` introduces `Re-DMD`, which explicitly **rewards motion dynamics** during distillation.
*   **vs. Other RL/DPO Methods**: Most RL-based alignment methods involve a complex, separate fine-tuning stage that can be unstable. `Re-DMD` elegantly integrates the reward signal as a simple **weighting factor within the existing DMD loss**. This is computationally cheaper, more stable, and avoids backpropagating through the reward model, which can be noisy.

# 4. Methodology
## 4.1. Principles
The core principle of `Reward Forcing` is to directly tackle the two primary weaknesses of prior streaming video models:
1.  **Context Brittleness:** The model's "memory" (the `KV cache`) is either too short (standard sliding window) or too rigid (static attention sink). The paper's solution is to create a flexible, evolving memory with **`EMA-Sink`**.
2.  **Motion Blindness:** The training objective (`DMD`) does not differentiate between good and bad motion. The paper's solution is to make the objective motion-aware by using a reward to "force" it to prioritize dynamics, leading to **`Re-DMD`**.

    These two components work together within the `Self-Forcing` autoregressive training loop.

The overall architecture is shown in the figure below. It depicts the autoregressive generation process, where the model generates video chunks. The `EMA-Sink` mechanism updates the KV cache, while the `Re-DMD` process uses a reward function to weight gradients from the teacher model to enhance motion.

![该图像是示意图，展示了奖励强制（Reward Forcing）框架的结构，包括动态摩托车生成过程。图中展示了当前关键值缓存（Current KV cache）的更新过程，以及生成视频的关键元素如教师梯度和奖励函数。相关公式包括 EMA 更新，标记为 `EMA ext{ update}`。](images/3.jpg)
*该图像是示意图，展示了奖励强制（Reward Forcing）框架的结构，包括动态摩托车生成过程。图中展示了当前关键值缓存（Current KV cache）的更新过程，以及生成视频的关键元素如教师梯度和奖励函数。相关公式包括 EMA 更新，标记为 `EMA ext{ update}`。*

## 4.2. Core Methodology In-depth
### 4.2.1. `EMA-Sink`: State Packaging for Long Video
The problem `EMA-Sink` solves is how to maintain a useful, long-term context for an autoregressive model without it becoming computationally expensive or overly rigid.

The figure below visually contrasts `EMA-Sink` with previous methods. (a) `Window Attention` quickly forgets the past. (b) `Sliding Window with attention sinks` holds onto the very first frame, causing over-reliance. (c) `EMA-Sink` continuously updates its summary of the past, providing a balanced context.

![Figure 2. Comparison of EMA Sink with Existing Methods. Long video generation models typically extrapolate beyond their training sequence length during inference. (a) Window Attention caches only recent tokens for efficient inference but suffers performance degradation. (b) Sliding Window with attention sinks retains initial tokens for stable attention computation and recent tokens for extrapolation. However, discarding intermediate frames causes over-reliance on the first frame, leading to "frame copying" and stiff transitions. (c) EMA Sink preserves full history through exponential moving average (EMA) updates of all historical frames, maintaining stable and consistent performance in long video extrapolation without increasing computational cost.](images/2.jpg)
*该图像是示意图，比较了三种长视频生成方法的注意力机制。左侧是窗口注意力，性能下降；中间是带有注意力沉 sink 的滑动窗口，导致对第一帧过度依赖；右侧是我们提出的 EMA-Sink，利用 EMA 更新实现历史帧的保留，保持稳定的性能而不增加计算成本。*

The `EMA-Sink` mechanism works as follows during the autoregressive generation process:

1.  **Setup:** The model uses a sliding attention window of size $w$. The `KV cache` stores the key and value vectors for the frames in this window. Additionally, a small, fixed-size set of "sink" tokens, represented by their key and value vectors ($S_K$ and $S_V$), is maintained. These are initialized using the very first frames of the video.

2.  **Eviction and Fusion:** As the generation proceeds and the sliding window moves forward, the oldest frame in the window is "evicted." For example, when generating frame $i$, frame `i-w` is pushed out of the local context. Instead of being discarded, its key-value pair, $(K^{i-w}, V^{i-w})$, is fused into the sink tokens using an exponential moving average (EMA).

3.  **Update Rule:** The update is performed using the following formulas:
    \$
    { \pmb { S } } _ { K } ^ { i } = \alpha \cdot { \pmb { S } } _ { K } ^ { i - 1 } + ( 1 - \alpha ) \cdot { \pmb { K } } ^ { i - w }
    \$
    \$
    { \pmb { S } } _ { V } ^ { i } = \alpha \cdot { \pmb { S } } _ { V } ^ { i - 1 } + ( 1 - \alpha ) \cdot { \pmb { V } } ^ { i - w }
    \$
    *   $\pmb{S}_K^i, \pmb{S}_V^i$: The key and value sink states after processing frame $i$.
    *   $\pmb{S}_K^{i-1}, \pmb{S}_V^{i-1}$: The previous sink states.
    *   $\pmb{K}^{i-w}, \pmb{V}^{i-w}$: The key and value vectors of the frame being evicted from the window.
    *   $\alpha$: A momentum decay factor (e.g., 0.99). A high $\alpha$ means the sink state changes slowly, retaining a long memory. A lower $\alpha$ means it updates more quickly with recent information. This single hyperparameter controls the balance between long-term and short-term context.

4.  **Attention Computation:** When generating the current frame, the model's attention mechanism is given access to both the local window context and the updated global sink tokens. The final key and value sets for the attention calculation are formed by prepending the sink states to the current window's states:
    \$
    K_{\mathrm{global}}^{i} = \left[ S_{K}^{i} ; K^{i-w+1:i} \right]
    \$
    \$
    V_{\mathrm{global}}^{i} = \left[ S_{V}^{i} ; V^{i-w+1:i} \right]
    \$
    This allows every new frame to attend to a compressed summary of the *entire* video history ($S_K$, $S_V$) as well as the fine-grained details of the most recent frames. This breaks the information bottleneck of a fixed window size without increasing computational cost, as the sink state is of constant size.

### 4.2.2. Rewarded Distribution Matching Distillation (`Re-DMD`)
The problem `Re-DMD` solves is the motion-blindness of standard `DMD`. The goal is to bias the distillation process to favor generated samples that exhibit high motion quality.

The method is derived from a reinforcement learning principle called **Reward-Weighted Regression (RWR)**, which frames RL as an Expectation-Maximization (EM) problem.

1.  **The RL Objective:** The starting point is a general RL objective that trades off maximizing a reward $r$ against staying close to a prior distribution $q$:
    \$
    \mathcal { T } _ { \mathrm { RL } } ( p , q ) = \mathbb { E } \Big [ \frac { r ( { \pmb x } _ { 0 } , { \pmb c } ) } { \beta } - \log \frac { p ( { \pmb x } _ { 0 } | { \pmb c } ) } { q ( { \pmb x } _ { 0 } | { \pmb c } ) } \Big ]
    \$
    *   $p(\pmb{x}_0|\pmb{c})$: The new, optimized policy (model distribution) we want to find.
    *   $q(\pmb{x}_0|\pmb{c})$: The original model distribution we want to stay close to.
    *   $r(\pmb{x}_0, \pmb{c})$: The reward for a generated sample $\pmb{x}_0$ given condition $\pmb{c}$. In this paper, this is the motion quality score from a VLM.
    *   $\beta$: A temperature parameter that controls the strength of the reward. A smaller $\beta$ means the reward has a stronger influence.

2.  **E-Step (Finding the Optimal Distribution):** The theoretical optimal distribution $p$ that maximizes this objective is an exponentially weighted version of the original distribution $q$:
    \$
    p ( \pmb { x } _ { 0 } | \pmb { c } ) = \frac { 1 } { Z ( \pmb { c } ) } q ( \pmb { x } _ { 0 } | \pmb { c } ) \exp \Big ( \frac { r ( \pmb { x } _ { 0 } , \pmb { c } ) } { \beta } \Big )
    \$
    *   $Z(\pmb{c})$ is a normalization constant (partition function) that is intractable to compute.

3.  **M-Step (Projecting back to the Model):** Now, we want to train our parametric student model, whose distribution is $p_{fake}$, to match this ideal rewarded distribution. The paper cleverly embeds this into the `DMD` framework. The final `Re-DMD` objective becomes:
    \$
    \mathcal { T } _ { \mathrm { Re - DMD } } = \mathbb { E } _ { p ( c ) p _ { \mathrm { f a k e } } ^ { \prime } ( \pmb { x } _ { 0 } | c ) } \left[ \frac { \exp \left( r \left( \pmb { x } _ { 0 } , \pmb { c } \right) / \beta \right) } { Z ( \pmb { c } ) } \log \frac { p _ { \mathrm { f a k e } } \left( \pmb { x } _ { 0 } | \pmb { c } \right) } { p _ { \mathrm { r e a l } } \left( \pmb { x } _ { 0 } | \pmb { c } \right) } \right]
    \$
    This looks complicated, but its gradient has a very simple and elegant form.

4.  **The `Re-DMD` Gradient:** The key insight is that when we compute the gradient with respect to the student generator's parameters $\theta$, the intractable partition function $Z(\pmb{c})$ disappears. The final, practical gradient used for training is:
    \$
    \nabla_{\theta} \mathcal{J}_{\mathrm{Re-DMD}} \approx - \mathbb{E}_{t} \Big( \int \exp(r^{c}(\pmb{x}_{t}) / \beta) \cdot \big( s_{\mathrm{real}}(\Psi(G_{\theta}(\epsilon), t), t) - s_{\mathrm{fake}}\big(\Psi(G_{\theta}(\epsilon), t), t) \big) \frac{\mathrm{d}G_{\theta}(\epsilon)}{\mathrm{d}\theta} \mathrm{d}\epsilon \Big)
    \$
    *   **Comparison to standard DMD:** This is almost identical to the standard DMD gradient (Eq. 1), with one crucial addition: the weighting factor $\exp(r^c(\pmb{x}_t) / \beta)$.
    *   **Intuition:** For each generated sample $G_{\theta}(\epsilon)$, the model computes its motion reward $r$. If the reward is high (good motion), the exponential term is large, and this sample's contribution to the gradient update is **amplified**. If the reward is low (static video), the term is small, and its contribution is **suppressed**.
    *   **Efficiency:** This is very efficient because the reward $r$ is treated as a fixed weight. There is no need to compute gradients of the reward or backpropagate through the reward model (the VLM), which avoids a major source of instability and computational overhead in traditional RL fine-tuning.

# 5. Experimental Setup
## 5.1. Datasets
*   **Training Data:** The model was trained on **`VidProM`**, a large-scale dataset containing millions of real prompt-gallery pairs for text-to-video models. The authors used a filtered and LLM-augmented version with 16k ODE solution pairs sampled from the base model for distillation.
*   **Evaluation Data:**
    *   For short video generation, they used 946 prompts from the official **`VBench`** benchmark, which were rewritten using a large language model (`Qwen`) for clarity.
    *   For long video generation, they used the first 128 prompts from **`MovieGen`**, following the setup of `CausVid`.

## 5.2. Evaluation Metrics
### 5.2.1. VBench & VBench-Long
*   **Conceptual Definition:** `VBench` is a comprehensive benchmark suite designed to evaluate video generative models across multiple dimensions. `VBench-Long` is an extension specifically for long videos. It covers two main aspects:
    1.  **Video Quality:** Assesses perceptual quality, consistency, and smoothness. Dimensions include `Subject Consistency`, `Background Consistency`, `Motion Smoothness`, `Dynamic Degree`, `Aesthetic Quality`, and `Imaging Quality`.
    2.  **Semantic Fidelity:** Measures how well the video content aligns with the input text prompt. Dimensions include `Object Class`, `Human Action`, `Color`, `Spatial Relationship`, etc.
*   **Mathematical Formula:** `VBench` uses a variety of specialized models and algorithms to calculate scores for each dimension. The final "Total Score" is a weighted average of these normalized sub-scores. The paper does not provide a single formula, as it is a complex suite of metrics.

### 5.2.2. Drift
*   **Conceptual Definition:** The `Drift` metric quantifies the temporal inconsistency in a long video. It measures how much the visual quality fluctuates over time. A low drift score indicates that the video maintains a stable quality from beginning to end, while a high drift score suggests quality degradation or significant changes over time.
*   **Mathematical Formula:** As provided in the supplementary material, the drift for a single video $V_i$ is calculated as the standard deviation of the `Imaging Quality` scores across $M$ short clips segmented from the long video.
    \$
    \operatorname { D r i f t } ( V _ { i } ) = { \sqrt { \frac { 1 } { M - 1 } \sum _ { j = 1 } ^ { M } ( s _ { i , j } - { \bar { s } } _ { i } ) } }
    \$
*   **Symbol Explanation:**
    *   $s_{i,j}$: The `Imaging Quality` score (from `VBench`) of the $j$-th clip of video $i$.
    *   $\bar{s}_i$: The average `Imaging Quality` score across all clips of video $i$.
    *   $M$: The number of clips the video is divided into (30 in this paper).

### 5.2.3. Qwen3-VL Score
*   **Conceptual Definition:** This is an automated evaluation using a powerful Vision-Language Model (`Qwen3-VL-235B-A22B-Instruct`) to score generated videos on a scale of 1 (Poor) to 5 (Perfect). It assesses three key aspects: `Text Alignment` (does it match the prompt?), `Dynamics` (is the motion fluid and natural?), and `Visual Quality` (is it clear and artifact-free?). This provides a proxy for human judgment.

### 5.2.4. FPS (Frames Per Second)
*   **Conceptual Definition:** FPS measures the inference speed of the model. It is the number of video frames the model can generate in one second. A higher FPS is better, with values above ~24 FPS considered "real-time."
*   **Mathematical Formula:**
    \$
    \text{FPS} = \frac{\text{Total Frames Generated}}{\text{Total Time Taken (seconds)}}
    \$

## 5.3. Baselines
The paper compares `Reward Forcing` against a comprehensive set of recent open-source video generation models, including:
*   **Full Diffusion Models:** `LTX-Video`, `Wan-2.1` (the teacher model).
*   **Autoregressive Models:** `SkyReels-V2`, `MAGI-1`, `NOVA`, `Pyramid Flow`.
*   **Distilled Autoregressive Models (Direct Competitors):** `CausVid`, `Self Forcing`, `LongLive`, `Rolling Forcing`.
    This is a strong set of baselines as it covers both the high-quality (but slow) teacher and the most relevant state-of-the-art efficient generation methods.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### 6.1.1. Short Video Generation
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Params</th>
<th rowspan="2">FPS↑</th>
<th colspan="3">VBench evaluation scores ↑</th>
</tr>
<tr>
<th>Total</th>
<th>Quality</th>
<th>Semantic</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6"><strong>Diffusion</strong></td>
</tr>
<tr>
<td>LTX-Video [13]</td>
<td>1.9B</td>
<td>8.98</td>
<td>80.00</td>
<td>82.30</td>
<td>70.79</td>
</tr>
<tr>
<td>Wan-2.1 [72]</td>
<td>1.3B</td>
<td>0.78</td>
<td><u>84.26</u></td>
<td><u>85.30</u></td>
<td>80.09</td>
</tr>
<tr>
<td colspan="6"><strong>Autoregressive</strong></td>
</tr>
<tr>
<td>SkyReels-V2 [7]</td>
<td>1.3B</td>
<td>0.49</td>
<td>82.67</td>
<td>84.70</td>
<td>74.53</td>
</tr>
<tr>
<td>MAGI-1 [69]</td>
<td>4.5B</td>
<td>0.19</td>
<td>79.18</td>
<td>82.04</td>
<td>67.74</td>
</tr>
<tr>
<td>NOVA [13]</td>
<td>0.6B</td>
<td>0.88</td>
<td>80.12</td>
<td>80.39</td>
<td>79.05</td>
</tr>
<tr>
<td>Pyramid Flow [33]</td>
<td>2B</td>
<td>6.7</td>
<td>81.72</td>
<td>84.74</td>
<td>69.62</td>
</tr>
<tr>
<td>CausVid [89]</td>
<td>1.3B</td>
<td>17.0</td>
<td>82.88</td>
<td>83.93</td>
<td>78.69</td>
</tr>
<tr>
<td>Self Forcing [30]</td>
<td>1.3B</td>
<td>17.0</td>
<td>83.80</td>
<td>84.59</td>
<td>80.64</td>
</tr>
<tr>
<td>LongLive [82]</td>
<td>1.3B</td>
<td>20.7</td>
<td>83.22</td>
<td>83.68</td>
<td><u>81.37</u></td>
</tr>
<tr>
<td>Rolling Forcing [45]</td>
<td>1.3B</td>
<td>17.5</td>
<td>81.22</td>
<td>84.08</td>
<td>69.78</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td><strong>1.3B</strong></td>
<td><strong>23.1</strong></td>
<td><strong>84.13</strong></td>
<td><strong>84.84</strong></td>
<td><strong>81.32</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Quality:** `Reward Forcing` (Ours) achieves a total `VBench` score of 84.13, nearly matching the powerful but extremely slow teacher model `Wan-2.1` (84.26) and outperforming all other autoregressive baselines.
*   **Speed:** The most striking result is the FPS. At **23.1 FPS**, it is by far the fastest model, achieving a significant speedup over competitors like `LongLive` (20.7 FPS) and `Self Forcing` (17.0 FPS). This demonstrates the framework's efficiency.
*   **Efficiency-Quality Tradeoff:** The model successfully balances high quality and high speed, a key goal of the research.

### 6.1.2. Long Video Generation
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="7">VBench Long Evaluation Scores ↑</th>
<th rowspan="2">Drift↓</th>
<th colspan="3">Qwen3-VL Score ↑</th>
</tr>
<tr>
<th>Total</th>
<th>Subject</th>
<th>Background</th>
<th>Smoothness</th>
<th>Dynamic</th>
<th>Aesthetic</th>
<th>Imaging Quality</th>
<th>Visual</th>
<th>Dynamic</th>
<th>Text</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="12">Diffusion Forcing</td>
</tr>
<tr>
<td>SkyReels-V2 [7]</td>
<td>75.94</td>
<td>96.43</td>
<td>96.59</td>
<td>98.91</td>
<td>39.86</td>
<td>50.76</td>
<td>58.65</td>
<td>7.315</td>
<td>3.30</td>
<td>3.05</td>
<td>2.70</td>
</tr>
<tr>
<td colspan="12">Distilled Causal</td>
</tr>
<tr>
<td>Caus Vid [89]</td>
<td>77.78</td>
<td>97.92</td>
<td>96.62</td>
<td>98.47</td>
<td>27.55</td>
<td>58.39</td>
<td>67.77</td>
<td>2.906</td>
<td>4.66</td>
<td>3.16</td>
<td>3.32</td>
</tr>
<tr>
<td>Self Forcing [30]</td>
<td>79.34</td>
<td>97.10</td>
<td>96.03</td>
<td>98.48</td>
<td>54.94</td>
<td>54.40</td>
<td>67.61</td>
<td>5.075</td>
<td>3.89</td>
<td>3.44</td>
<td>3.11</td>
</tr>
<tr>
<td>LongLive [82]</td>
<td>79.53</td>
<td>97.96</td>
<td>96.50</td>
<td>98.79</td>
<td>35.54</td>
<td>57.81</td>
<td>69.91</td>
<td>2.531</td>
<td>4.79</td>
<td>3.81</td>
<td>3.98</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td><strong>81.41</strong></td>
<td><strong>97.26</strong></td>
<td><strong>96.05</strong></td>
<td><strong>98.88</strong></td>
<td><strong>66.95</strong></td>
<td><strong>57.47</strong></td>
<td><strong>70.06</strong></td>
<td><strong>2.505</strong></td>
<td><strong>4.82</strong></td>
<td><strong>4.18</strong></td>
<td><strong>4.04</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Dynamics:** This is where the model truly shines. It achieves a `Dynamic` score of **66.95**, a massive improvement over the next best, `Self Forcing` (54.94), and nearly double that of `LongLive` (35.54). This directly validates the effectiveness of the `Re-DMD` component. The qualitative results in Figure 4 also show this, with "Ours" producing visibly more dynamic motion.
*   **Consistency:** The model achieves the lowest `Drift` score (**2.505**), indicating superior temporal consistency and minimal quality degradation over the 60-second generation. This validates the effectiveness of the `EMA-Sink` mechanism in preventing error accumulation without sacrificing dynamics.
*   **Overall Performance:** `Reward Forcing` achieves the highest `Total Score` (81.41) on `VBench-Long` and the highest scores across all three dimensions of the `Qwen3-VL` evaluation, confirming its state-of-the-art performance in long video generation. Figure 5 demonstrates this superior long-term consistency.

    ![该图像是一个示意图，展示了不同视频生成方法在特定场景下的效果对比。第一行为本研究提出的策略，后续行分别展示了 Long Live、Self Forcing、CausVid 和 SkyReels V2 方法生成的视频帧，展示了不同时间点上图像的运动效果。](images/4.jpg)
    *Figure 4 from the paper, showing enhanced motion dynamics from "Ours" compared to baselines.*

    ![该图像是一个示意图，展示了不同视频生成方法在预定时间间隔内的输出效果，包含了'我们的方法'、'Long Live'、'Self Forcing'、'CausVid'和'SkyReels V2'等处理结果，展示了各方法在0到60秒的变化。](images/5.jpg)
    *Figure 5 from the paper, showing superior temporal consistency from "Ours" over 60 seconds.*

## 6.2. Ablation Studies / Parameter Analysis
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4">VBench Evaluation Scores ↑</th>
<th rowspan="2">Drift↓</th>
</tr>
<tr>
<th>Background</th>
<th>Smoothness</th>
<th>Dynamic</th>
<th>Quality</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6"><strong>Improvement</strong></td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td>95.07</td>
<td>98.82</td>
<td><strong>64.06</strong></td>
<td>70.57</td>
<td>2.51</td>
</tr>
<tr>
<td>w/o Re-DMD</td>
<td>95.85</td>
<td>98.91</td>
<td>43.75</td>
<td>71.42</td>
<td><strong>1.77</strong></td>
</tr>
<tr>
<td>w/o EMA</td>
<td>95.61</td>
<td>98.64</td>
<td>35.15</td>
<td>70.50</td>
<td>2.65</td>
</tr>
<tr>
<td>w/o Sink</td>
<td>94.94</td>
<td>98.56</td>
<td>51.56</td>
<td>69.92</td>
<td>5.08</td>
</tr>
<tr>
<td colspan="6"><strong>Impact of α</strong></td>
</tr>
<tr>
<td>α = 0.99</td>
<td>95.90</td>
<td>98.96</td>
<td>65.15</td>
<td>70.81</td>
<td>2.52</td>
</tr>
<tr>
<td>α = 0.9</td>
<td>95.80</td>
<td>99.09</td>
<td>63.15</td>
<td>71.37</td>
<td>3.23</td>
</tr>
<tr>
<td>α = 0.5</td>
<td>94.57</td>
<td>98.89</td>
<td>64.37</td>
<td>71.11</td>
<td>3.78</td>
</tr>
<tr>
<td colspan="6"><strong>Impact of β</strong></td>
</tr>
<tr>
<td>β = 1</td>
<td>95.14</td>
<td>98.31</td>
<td>54.68</td>
<td>71.73</td>
<td>2.63</td>
</tr>
<tr>
<td>β = 2/3</td>
<td>95.02</td>
<td>98.46</td>
<td>60.93</td>
<td>70.61</td>
<td>1.91</td>
</tr>
<tr>
<td><strong>β = 1/2 (Ours)</strong></td>
<td>95.07</td>
<td>98.82</td>
<td><strong>64.06</strong></td>
<td>70.57</td>
<td>2.51</td>
</tr>
<tr>
<td>β = 1/3</td>
<td>94.94</td>
<td>98.43</td>
<td>58.59</td>
<td>69.29</td>
<td>2.02</td>
</tr>
<tr>
<td>β = 1/5</td>
<td>92.40</td>
<td>96.40</td>
<td>94.53</td>
<td>68.26</td>
<td>3.13</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Impact of `Re-DMD`:** Removing `Re-DMD` (`w/o Re-DMD` row) causes the `Dynamic` score to plummet from 64.06 to 43.75. This is the clearest evidence that `Re-DMD` is the key driver of the model's enhanced motion dynamics.
*   **Impact of `EMA-Sink`:** Comparing `w/o Re-DMD` to `w/o EMA` shows the effect of the EMA update rule itself. Removing it (`w/o EMA`) further degrades the `Dynamic` score (43.75 to 35.15) and reduces `Smoothness`. This confirms that the continuous fusion of recent information is crucial.
*   **Impact of Sinks entirely:** Removing the sink mechanism completely (`w/o Sink`) leads to a massive increase in `Drift` (from ~2.5 to 5.08) and a drop in overall `Quality`. This confirms that having a long-term context anchor (a sink) is essential for stability.
*   **Impact of $β$ (Reward Weight):** The $β$ parameter controls the trade-off between dynamics and fidelity. As $β$ gets smaller, the reward is weighted more heavily. An extremely small $β$ (1/5) leads to a massive `Dynamic` score (94.53) but severely degrades other metrics like `Background` consistency, `Smoothness`, and `Quality`. A large $β$ (1) results in an insufficient `Dynamic` score (54.68). The chosen value of $β = 1/2$ represents the optimal balance.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
The paper successfully introduces **`Reward Forcing`**, a novel and effective framework that addresses the critical problem of motion stagnation in efficient, streaming video generation. The authors' two key innovations, **`EMA-Sink`** and **`Re-DMD`**, work in tandem to overcome the limitations of prior autoregressive models. `EMA-Sink` provides a dynamic and computationally cheap solution for maintaining long-term coherence without being rigidly tied to initial frames. `Re-DMD` introduces a motion-aware training objective that effectively distills dynamic capabilities from a teacher model. The result is a model that achieves state-of-the-art performance in both video quality and motion dynamics, all while operating at real-time speeds. This work sets a new standard for generating dynamic and interactive virtual worlds efficiently.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations and areas for future research in the supplementary material:
*   **Reward Model Dependency:** The entire framework's ability to enhance "motion" is contingent on the reward model's definition of good motion. The current reward model (`VideoAlign`) may prioritize certain types of dynamics over others, and its judgment may not always align perfectly with human perception or other quality dimensions.
*   **Generalizability:** While the authors claim the method is a "plug-in-and-play" module, its performance may still be dependent on the architecture of the base model it is applied to. Further testing on a wider variety of base models would be needed to fully confirm its generalizability.
*   **Future Work:** The authors suggest that future work could involve designing more sophisticated reward models that incorporate a deeper understanding of real-world physics and semantic priors. This would allow the model to generate not just dynamic, but also more plausible and meaningful, motion.

## 7.3. Personal Insights & Critique
*   **Elegance in Simplicity:** The most impressive aspect of this paper is the elegance of its solutions. `Re-DMD` is particularly clever. Instead of implementing a complex and often unstable reinforcement learning pipeline, it integrates the reward signal as a simple, stable weighting factor in a well-understood loss function (`DMD`). This is a powerful design pattern for aligning generative models that is both effective and practical.
*   **Broad Applicability of `EMA-Sink`:** The `EMA-Sink` mechanism is a general solution for managing context in long-sequence autoregressive generation. While applied to video here, the same principle could be highly effective for other domains like long-form text generation, music generation, or any task involving streaming data with long-range dependencies.
*   **The "Reward" Frontier:** This work highlights a growing trend in generative AI: moving beyond simple pixel-level or distribution-level matching towards optimizing for higher-level, often semantic, qualities defined by reward models or human feedback. The main challenge, as noted by the authors, is the quality and potential bias of the reward models themselves. As vision-language models become more powerful and nuanced, frameworks like `Reward Forcing` will become even more effective.
*   **Critique on Evaluation:** While the paper uses strong benchmarks (`VBench`) and an automated VLM-based evaluation, a more extensive human user study would be beneficial to fully validate that the increased "Dynamic" score truly corresponds to a better perceptual experience. The supplementary material mentions a user study, which strengthens the claims, but this dependency on automated or limited human evaluation is a general challenge in the field. Overall, `Reward Forcing` is a strong piece of research that provides a clear and impactful contribution to a very active area of generative AI.