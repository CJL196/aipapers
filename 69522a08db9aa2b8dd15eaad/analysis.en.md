# 1. Bibliographic Information

## 1.1. Title
VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory

## 1.2. Authors
Yifei Vu (HKU), Xiaoshan Wu (HKU), Xinting Hu (HKU), Tao Hu (PICO, ByteDance), Yang-Tian Sun (HKU), Xiaoyang Lyu (HKU), Bo Wang (HKU), Lin Ma (PICO, ByteDance), Yuewen Ma (PICO, ByteDance), Zhongrui Wang (SUSTech), and Xiaojuan Qi (HKU).

## 1.3. Journal/Conference
This paper was published as a preprint on arXiv on December 4, 2024. Given the affiliations (HKU, ByteDance) and the state-of-the-art nature of the work, it is likely intended for a top-tier computer vision or machine learning conference such as CVPR, ICCV, or NeurIPS.

## 1.4. Publication Year
2024 (Preprint date: 2025-12-04T07:06:02.000Z).

## 1.5. Abstract
The paper addresses the challenge of maintaining coherence in long-horizon (minute-scale) video generation using **Autoregressive (AR) Diffusion**. Traditional AR models suffer from error accumulation and "content repetition" because they either lose context (sliding windows) or freeze it (attention sinks). The authors propose **VideoSSM**, which treats video synthesis as a recurrent dynamical process. It utilizes a **Hybrid State-Space Memory** consisting of:
1.  **Local Memory:** A sliding-window context for fine details and motion.
2.  **Global Memory:** A **State-Space Model (SSM)** that recurrently compresses evicted history into an evolving global state.
    VideoSSM achieves state-of-the-art temporal consistency and motion stability, scaling linearly with sequence length while supporting interactive, prompt-based control.

## 1.6. Original Source Link
*   **Official Link:** [https://arxiv.org/abs/2512.04519](https://arxiv.org/abs/2512.04519)
*   **PDF Link:** [https://arxiv.org/pdf/2512.04519v1.pdf](https://arxiv.org/pdf/2512.04519v1.pdf)

# 2. Executive Summary

## 2.1. Background & Motivation
Generating long, high-quality videos (over a minute) is a frontier in AI. Current **Diffusion Transformers (DiT)** typically generate short clips because their computational cost grows quadratically with video length. **Autoregressive (AR) models** solve this by generating frames one by one, but they face three major hurdles:
*   **Error Accumulation:** Small mistakes in early frames magnify over time.
*   **Motion Drift:** The subject or scene slowly "morphs" into something else.
*   **Content Repetition:** To stay stable, models often attend to the very first frames (attention sinks), which causes the video to loop or become static.

    The authors identify that what is missing is a **dynamic global memory**—something that updates as the scene changes, much like human memory, rather than just looking back at a few fixed "anchor" frames.

## 2.2. Main Contributions / Findings
*   **Hybrid Memory Architecture:** Combines a lossless sliding window (local) with a compressed, evolving State-Space Model (global).
*   **Linear Scalability:** Unlike standard attention, the complexity grows linearly with time, making hour-long generation theoretically feasible.
*   **Dynamic Consistency:** The model maintains the identity of subjects over 60+ seconds without freezing the motion, avoiding the "stagnation" seen in previous methods.
*   **Interactive Control:** The model can change the narrative mid-stream (prompt switching) while maintaining scene coherence.
*   **State-of-the-Art (SOTA) Performance:** Outperforms previous AR models like `Self-Forcing` and `LongLive` on benchmarks like VBench, particularly in motion stability and aesthetic quality.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Diffusion Transformers (DiT)
A **Diffusion Transformer** is a generative model that combines **Diffusion Models** (which create images by reversing a noise process) with **Transformers** (which use `self-attention` to model relationships between different parts of an input). In video, the Transformer treats patches of frames as "tokens."
*   **Self-Attention Formula:**
    \$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    \$
    Where $Q$ (Query), $K$ (Key), and $V$ (Value) represent different projections of the input data. $d_k$ is the dimensionality of the keys.

### 3.1.2. Autoregressive (AR) Generation
In AR generation, the model predicts the "next" item in a sequence based on all "previous" items. For video, it means generating frame $t+1$ by using frames `1` through $t$ as context. This is typically done using a **KV Cache** (Key-Value Cache) to store the history so it doesn't have to be recomputed.

### 3.1.3. State-Space Models (SSM)
An **SSM** is a type of sequence model (like `Mamba`) that summarizes the entire history into a single, fixed-size hidden state. Instead of looking at every past token (which is slow), it updates its "memory" state at every step. This makes it extremely fast and memory-efficient for long sequences.

## 3.2. Previous Works
*   **Self-Forcing [25]:** Introduced a training method where the model is trained on its own previous (potentially imperfect) predictions, helping it handle errors during long-term inference.
*   **LongLive [57]:** Used "Attention Sinks"—fixing the very first frames of a video in memory forever to act as a stable anchor. However, this often leads to scene looping.
*   **Wan 2.1 [45]:** A state-of-the-art bidirectional video model that VideoSSM uses as its "teacher" during training.

## 3.3. Technological Evolution
The field moved from **Bidirectional models** (generating all frames at once, limited length) to **AR models** (streaming, but unstable). Within AR, methods evolved from **Fixed Windows** (losing history) to **Attention Sinks** (static history). **VideoSSM** represents the next step: **Dynamic Compressed History**, using SSMs to balance stability with the ability to evolve.

# 4. Methodology

## 4.1. Principles
The core intuition of VideoSSM is to mimic the human memory system:
1.  **Working Memory (Local):** Highly detailed but has a limited capacity (the sliding window).
2.  **Long-term Memory (Global):** An abstract, compressed summary of everything that happened before (the SSM).

## 4.2. Core Methodology In-depth

### 4.2.1. Local Memory: Sliding Window Self-Attention
The model first processes the current frame tokens to extract features. For a current frame at time $t$, we compute Queries ($Q_t$), Keys ($K_t$), and Values ($V_t$) using learnable weight matrices $W_Q, W_K, W_V$:
\$
\{ \mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t \} = \{ \mathbf{H}_t^{\mathrm{in}} \mathbf{W}_Q, \mathbf{H}_t^{\mathrm{in}} \mathbf{W}_K, \mathbf{H}_t^{\mathrm{in}} \mathbf{W}_V \}
\$
Where $H_t^{\mathrm{in}}$ is the input hidden state. To keep memory usage constant, the model uses a sliding window of size $L$. The local KV cache stores only the $L$ most recent tokens:
\$
\mathbf{K}_t^{\mathrm{local}} = [\mathbf{K}_{\mathrm{sink}}, \mathbf{K}_{t-L+1} : \mathbf{K}_t], \quad \mathbf{V}_t^{\mathrm{local}} = [\mathbf{V}_{\mathrm{sink}}, \mathbf{V}_{t-L+1} : \mathbf{V}_t]
\$
The local output $H_t^{\mathrm{local}}$ is then computed using standard causal attention:
\$
\mathbf{H}_t^{\mathrm{local}} = \mathrm{SelfAttention}(\mathbf{Q}_t, \mathbf{K}_t^{\mathrm{local}}, \mathbf{V}_t^{\mathrm{local}})
\$

### 4.2.2. Global Memory: Dynamic State Computation
When a token moves out of the $L$-sized window, it is "evicted." Instead of being deleted, it is sent to the **Global Memory** module.

**Step 1: Synchronized Gate Caching**
Two gates control how information enters and leaves the global state: an **injection gate** $\beta_t$ (what to add) and a **decay gate** $\alpha_t$ (what to forget). They are calculated from the hidden state $H_t^{\mathrm{in}}$:
\$
\begin{array}{rl} & \beta_t = \sigma(\mathbf{W}_{\beta} \mathbf{H}_t^{\mathrm{in}}) \\ & \alpha_t = -\exp(\mathbf{A}) \cdot \mathrm{SoftPlus}(\mathbf{W}_{\alpha} \mathbf{H}_t^{\mathrm{in}} + \mathbf{B}) \end{array}
\$
*   $\sigma$: The Sigmoid function, squashing values between 0 and 1.
*   $\mathrm{SoftPlus}$: A smooth version of ReLU ($f(x) = \ln(1 + e^x)$).
*   $A, W_{\alpha}, W_{\beta}, B$: Learnable weights and biases.

**Step 2: State Update via Gated $\Delta$-rule**
The global state $M_t$ is updated using a "delta rule" that only stores the *unpredictable* part of new info, keeping the memory efficient:
\$
\begin{array}{rlr} & & \mathbf{V}_{\mathrm{new}, t}^{\mathrm{evt}} = \mathbf{V}_t^{\mathrm{evt}} - \mathrm{Predict}(\mathbf{M}_{t-1}, \mathbf{K}_t^{\mathrm{evt}}, \beta_t^{\mathrm{evt}}), \\ & & \mathbf{M}_t = \exp(\bar{\mathbf{g}}_t) \cdot \mathbf{M}_{t-1} + \mathbf{K}_t^{\mathrm{evt}} \cdot (\mathbf{V}_{\mathrm{new}, t}^{\mathrm{evt}})^T, \end{array}
\$
*   $M_t$: The global memory matrix at time $t$.
*   $\bar{g}_t$: The cumulative decay gate $\sum \alpha_s$, ensuring the memory doesn't explode.
*   $\mathrm{Predict}(\cdot)$: An estimation of what the model already "knows" about the new token based on previous states.

**Step 3: Retrieval**
To use this memory, the model "queries" the global state $M_t$ using the current $Q_t$:
\$
\begin{array}{r} \mathbf{g}_t^{\mathrm{out}} = \mathrm{Linear}(\mathbf{H}_t^{\mathrm{in}}), \quad \quad \quad \\ \mathbf{H}_t^{\mathrm{global}} = \mathrm{Swish}(\mathbf{g}_t^{\mathrm{out}} \odot \mathrm{RMSNorm}(\mathbf{Q}_t \mathbf{M}_t)) \end{array}
\$
*   $g_t^{\mathrm{out}}$: An output gate that decides how much global info to let through.
*   $\mathrm{RMSNorm}$: A normalization layer to keep signals stable.
*   $\mathrm{Swish}$: An activation function ($x \cdot \sigma(x)$).

### 4.2.3. Position-Aware Gated Fusion
Finally, the local and global outputs are combined. Early in the video, global memory is empty and unreliable, so the model should ignore it. The authors use a **position ratio** $\rho_t = (t+1)/T$ to compute a fusion gate $\gamma_t$:
\$
\gamma_t = \sigma(\mathbf{w}_{\mathrm{router}} \log(\rho_t) + b_{\mathrm{router}})
\$
As $t$ increases, $\gamma_t$ grows, allowing more global memory into the final state:
\$
\mathbf{H}_t^{\mathrm{fused}} = \mathbf{H}_t^{\mathrm{local}} + \gamma_t \cdot \mathbf{H}_t^{\mathrm{global}}
\$

The following figure (Figure 5 from the original paper) illustrates this complete hybrid architecture:

![Figure 5. Architecture of the proposed hybrid memory module. The input $H _ { t } ^ { \\mathrm { i n } }$ is processed in two streams. The local path (top) uses windowed attention with a sliding KV cache to compute $H _ { t } ^ { \\mathrm { l o c a l } }$ To pSaMoSM) to recurrently compress historical information into a memory state $M$ , which is retrieved to produce $H _ { t } ^ { \\mathrm { g l o b a l } }$ . A router then dynamically fuses the local and global outputs.](images/5.jpg)
*该图像是示意图，展示了提出的混合记忆模块的架构。输入的隐藏状态 $H_t^{\mathrm{in}}$ 通过两个路径进行处理。局部路径使用窗口注意力和滑动KV缓存计算 $H_t^{\mathrm{local}}$，并通过 pSaMoSM 将历史信息压缩到记忆状态 $M_t$ 中，随后通过回忆产生 $H_t^{\mathrm{global}}$。路由器动态融合局部和全局输出。*

## 4.3. Training Strategy
VideoSSM uses a two-stage distillation process:
1.  **Causal Distillation:** The model learns to mimic a powerful bidirectional "Teacher" (Wan 2.1) on short 5-second clips.
2.  **Long Video Training:** The model performs "Self-Rollouts"—it generates 60 seconds of video using its own predictions. It then uses **DMD (Distribution Matching Distillation)** loss to correct errors by comparing short windows of its long generation back to the high-quality Teacher.

# 5. Experimental Setup

## 5.1. Datasets
*   **VidProM:** A dataset containing millions of real text-to-video prompts. The authors used a filtered and LLM-extended version for training.
*   **VBench:** The primary evaluation suite. It contains various categories (Subject Consistency, Motion Smoothness, etc.) and provides prompts for both short (5s) and long (60s) video testing.

## 5.2. Evaluation Metrics

### 5.2.1. VBench Dimensions
For video quality, VBench uses several specialized metrics:
1.  **Subject Consistency:** Measures if the main character/object stays the same throughout the video.
2.  **Background Consistency:** Measures if the scenery stays stable.
3.  **Dynamic Degree:** Quantifies the amount of motion in the video (to check if the video has become a static image).
4.  **Temporal Flickering:** Quantifies the visual smoothness between adjacent frames.

### 5.2.2. Mathematical Foundation for Consistency
Most consistency metrics in VBench are calculated using **Cosine Similarity** between feature embeddings (usually from a `CLIP` or `DINO` model) across frames.
\$
\text{Similarity}(f_i, f_j) = \frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}
\$
*   $f_i, f_j$: Feature vectors of frame $i$ and frame $j$.
*   A higher score indicates better consistency.

## 5.3. Baselines
The authors compare VideoSSM against:
*   **Short-range AR models:** `CausVid`, `Self-Forcing`.
*   **Long-range AR models:** `LongLive` (attention sinks), `Rolling Forcing`.
*   **Commercial/Strong Open-Source:** `SkyReels-V2`, `Ltx-Video`, `Wan2.1`.

# 6. Results & Analysis

## 6.1. Core Results Analysis
VideoSSM demonstrates a superior balance between **stability** and **dynamics**. While `LongLive` achieves high consistency, it often results in static videos (low Dynamic Degree). VideoSSM achieves higher consistency *and* higher motion (Dynamic Degree of 50.50 vs LongLive's 37.50).

## 6.2. Data Presentation (Tables)

The following are the results from Table 1 of the original paper, comparing VideoSSM on short-video benchmarks:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">#Params</th>
<th colspan="3">Evaluation scores ↑</th>
</tr>
<tr>
<th>Total</th>
<th>Quality</th>
<th>Semantic</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Bidirectional Diffusion models</strong></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>LTX-Video [20]</td>
<td>1.9B</td>
<td>80.00</td>
<td>82.30</td>
<td>70.79</td>
</tr>
<tr>
<td>Wan2.1 [45]</td>
<td>1.3B</td>
<td>84.26</td>
<td>85.30</td>
<td>80.09</td>
</tr>
<tr>
<td><strong>Autoregressive models</strong></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>SkyReels-V2 [7]</td>
<td>1.3B</td>
<td>82.67</td>
<td>84.70</td>
<td>74.53</td>
</tr>
<tr>
<td>MAGI-1 [42]</td>
<td>4.5B</td>
<td>79.18</td>
<td>82.04</td>
<td>67.74</td>
</tr>
<tr>
<td>Caus Vid [60]</td>
<td>1.3B</td>
<td>81.20</td>
<td>84.05</td>
<td>69.80</td>
</tr>
<tr>
<td>NOVA [13]</td>
<td>0.6B</td>
<td>80.12</td>
<td>80.39</td>
<td>79.05</td>
</tr>
<tr>
<td>Pyramid Flow [27]</td>
<td>2B</td>
<td>81.72</td>
<td>84.74</td>
<td>69.62</td>
</tr>
<tr>
<td>Self Forcing [25]</td>
<td>1.3B</td>
<td>83.00</td>
<td>83.71</td>
<td>80.14</td>
</tr>
<tr>
<td>LongLive [57]</td>
<td>1.3B</td>
<td>83.52</td>
<td>84.26</td>
<td>80.53</td>
</tr>
<tr>
<td>Self Forcing ++ [12]</td>
<td>1.3B</td>
<td>83.11</td>
<td>83.79</td>
<td>80.37</td>
</tr>
<tr>
<td>Rolling Forcing [31]</td>
<td>1.3B</td>
<td>81.22</td>
<td>84.08</td>
<td>69.78</td>
</tr>
<tr>
<td>VideoSSM (Ours)</td>
<td>1.4B</td>
<td><strong>83.95</strong></td>
<td><strong>84.88</strong></td>
<td>80.22</td>
</tr>
</tbody>
</table>

The following are the results from Table 2, focusing on 60-second long video performance:

<table>
<thead>
<tr>
<th>Metric</th>
<th>LongLive</th>
<th>Self Forcing</th>
<th>VideoSSM (Ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Temporal Flickering ↑</td>
<td>97.86</td>
<td>97.24</td>
<td>97.70</td>
</tr>
<tr>
<td>Subject Consistency ↑</td>
<td>88.25</td>
<td>91.09</td>
<td><strong>92.51</strong></td>
</tr>
<tr>
<td>Background Consistency ↑</td>
<td>91.73</td>
<td>93.23</td>
<td><strong>93.95</strong></td>
</tr>
<tr>
<td>Motion Smoothness ↑</td>
<td>98.67</td>
<td>98.38</td>
<td>98.60</td>
</tr>
<tr>
<td>Dynamic Degree ↑</td>
<td>35.00</td>
<td>37.50</td>
<td><strong>50.50</strong></td>
</tr>
<tr>
<td>Aesthetic Quality ↑</td>
<td>60.02</td>
<td>55.74</td>
<td><strong>60.45</strong></td>
</tr>
</tbody>
</table>

## 6.3. User Study Results
The authors conducted a study with 40 participants ranking 32 videos. VideoSSM was preferred in 41.07% of cases, achieving an average rank of 1.85 (where 1 is best).

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th>Rank 1 (%)</th>
<th>Rank 2 (%)</th>
<th>Rank 3 (%)</th>
<th>Rank 4 (%)</th>
<th>Avg Rank</th>
</tr>
</thead>
<tbody>
<tr>
<td>Self Forcing</td>
<td>11.79</td>
<td>13.21</td>
<td>23.21</td>
<td>51.79</td>
<td>3.18</td>
</tr>
<tr>
<td>CausVid</td>
<td>7.50</td>
<td>16.07</td>
<td>42.14</td>
<td>34.29</td>
<td>3.03</td>
</tr>
<tr>
<td>LongLive</td>
<td>39.64</td>
<td>36.43</td>
<td>15.00</td>
<td>8.93</td>
<td>1.92</td>
</tr>
<tr>
<td>Ours</td>
<td><strong>41.07</strong></td>
<td>34.29</td>
<td>19.64</td>
<td>5.00</td>
<td><strong>1.85</strong></td>
</tr>
</tbody>
</table>

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
VideoSSM successfully unifies **AR Diffusion** with **Hybrid State-Space Memory**. By using an SSM to handle global context, it avoids the quadratic complexity of full attention while preventing the "freezing" effect of static attention sinks. It establishes a new benchmark for **minute-scale coherence** and **dynamic realism** in autoregressive video generation.

## 7.2. Limitations & Future Work
*   **Prompt Complexity:** While it handles prompt switching, highly complex or conflicting multi-modal conditions (like audio-sync) were not explored.
*   **3D Priors:** The memory is currently in the latent space; adding explicit 3D/geometric priors could further improve world-model consistency for robotics or gaming.
*   **Scaling:** Future work could extend this to even larger parameter models (e.g., 10B+ parameters) and much longer durations (hours).

## 7.3. Personal Insights & Critique
This paper is a clever application of **Mamba/SSM** technology to the specific "memory leak" problem in video generation. The most impressive part is the **Gated $\Delta$-rule**, which effectively filters "useless" information from the history to keep the memory state representative. 

However, one potential area for critique is the **distillation dependency**. The model is heavily reliant on the quality of the "Teacher" model (Wan 2.1). If the teacher has inherent biases or artifacts, VideoSSM will likely inherit them. Furthermore, while it handles 60-second videos well, the paper lacks a stress test for truly "infinite" generation (e.g., 1 hour), which would truly prove the stability of the SSM state over extreme horizons. Nonetheless, it is a significant step forward for interactive and streaming video AI.