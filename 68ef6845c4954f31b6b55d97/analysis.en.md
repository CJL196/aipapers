# 1. Bibliographic Information

*   **Title:** Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
*   **Authors:** Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman.
*   **Affiliations:** Adobe Research and The University of Texas at Austin.
*   **Journal/Conference:** This paper is a preprint available on arXiv. As of its publication date, it has not yet been published in a peer-reviewed conference or journal, which is a common practice for fast-moving fields like machine learning.
*   **Publication Year:** 2025 (as cited in the paper, likely submitted for a 2025 conference). The first version was submitted to arXiv in June 2024.
*   **Abstract:** The paper introduces `Self Forcing`, a new training method for autoregressive video diffusion models designed to solve the problem of **exposure bias**. This bias occurs because models are trained on perfect, ground-truth data but must generate sequences using their own imperfect outputs during inference. `Self Forcing` mimics the inference process during training by having the model generate each frame conditioned on its own previously generated frames. This allows for a holistic, video-level loss function to be applied, evaluating the entire generated sequence. To maintain efficiency, the method uses a few-step diffusion model and a special gradient truncation strategy. The authors also propose a `rolling KV cache` for efficient, long video generation. The resulting model achieves real-time, low-latency video generation that matches or exceeds the quality of slower, non-causal models.
*   **Original Source Link:** https://arxiv.org/abs/2506.08009

# 2. Executive Summary

*   **Background & Motivation (Why):**
    *   **Core Problem:** High-quality video generation models, especially those based on diffusion transformers, typically process all frames at once (bidirectionally). This makes them unsuitable for real-time applications like streaming or interactive content, where frames must be generated sequentially. Autoregressive (AR) models, which generate frame-by-frame, are a natural fit for these applications but suffer from a critical flaw: **exposure bias**.
    *   **Gap in Prior Work:** Existing AR training methods like `Teacher Forcing` (TF) and `Diffusion Forcing` (DF) train the model to predict the next frame using ground-truth past frames (either clean or noisy). During inference, however, the model must rely on its *own* previously generated frames, which are imperfect. This mismatch between the training distribution and the inference distribution causes errors to accumulate, leading to a degradation in video quality over time (e.g., color shifts, loss of coherence).
    *   **Fresh Angle:** This paper proposes to eliminate the train-test mismatch entirely. Instead of feeding the model ground-truth data, `Self Forcing` forces the model to perform a full autoregressive "rollout" during training. It learns from its own mistakes, making it robust to error accumulation and closing the gap caused by exposure bias.

*   **Main Contributions / Findings (What):**
    *   **Self Forcing (SF) Paradigm:** A novel training algorithm for AR video diffusion models where the model is conditioned on its own outputs during training, directly mirroring the inference process.
    *   **Holistic Video-Level Optimization:** SF enables the use of distribution-matching losses (like DMD, SiD, or GANs) on the entire generated video sequence, aligning the overall output distribution with the real data distribution, rather than just optimizing per-frame prediction accuracy.
    *   **Efficient Implementation:** The paper demonstrates that this seemingly expensive sequential training process can be made highly efficient through the use of a few-step diffusion backbone and a stochastic gradient truncation strategy. Counter-intuitively, it is shown to be more efficient than parallel training alternatives.
    *   **Rolling KV Cache Mechanism:** An efficient method for generating arbitrarily long videos by maintaining a fixed-size cache of past frame information, which avoids redundant computations and visual artifacts.
    *   **State-of-the-Art Performance:** The proposed model achieves real-time streaming video generation (17 FPS with sub-second latency on a single GPU) with visual quality that is competitive with, or even superior to, much slower bidirectional diffusion models.

# 3. Prerequisite Knowledge & Related Work

*   **Foundational Concepts:**
    *   **Diffusion Models:** A class of generative models that learn to create data by reversing a noise-adding process. They start with pure noise and iteratively refine it into a coherent sample (like an image or video frame) by predicting and removing the noise at each step. They are known for generating very high-quality samples.
    *   **Autoregressive (AR) Models:** Models that generate data sequentially. For a video, this means generating frame 1, then generating frame 2 conditioned on frame 1, then frame 3 conditioned on frames 1 and 2, and so on. This causal structure is ideal for streaming.
    *   **Exposure Bias:** A fundamental problem in training sequential models. The model is "exposed" only to perfect, ground-truth sequences during training. At test time, when it must generate a sequence from scratch, its first small error can lead to a cascade of larger errors, as it enters states it never saw during training.
    *   **Key-Value (KV) Caching:** An optimization technique used in Transformer models during autoregressive generation. To generate a new token (or frame), the model needs to attend to all previous tokens. Instead of recomputing the "key" (K) and "value" (V) matrices for all previous tokens at every step, they are cached and reused, dramatically speeding up inference.
    *   **Generative Adversarial Networks (GANs):** A framework where two neural networks, a **Generator** and a **Discriminator**, are trained in competition. The Generator tries to create realistic data, and the Discriminator tries to distinguish real data from generated data. This process pushes the Generator to produce increasingly realistic outputs.
    *   **Distribution Matching Distillation (DMD) & Score Identity Distillation (SiD):** Advanced techniques for training generative models. Instead of a simple pixel-wise loss, they aim to match the entire *distribution* of generated data to the distribution of real data. They often use a pre-trained "score" model to guide the training process.

*   **Previous Works & Differentiation:**
    The paper positions itself against three main training paradigms for AR video diffusion:

    1.  **Teacher Forcing (TF):** The model is trained to denoise the current frame ($x^i$) conditioned on the previous *clean, ground-truth* frames ($x^{<i}$). This is the classic approach but is highly susceptible to exposure bias.
    2.  **Diffusion Forcing (DF):** A more recent approach where the model denoises the current frame ($x^i$) conditioned on previous *noisy, ground-truth* frames ($x_{t^j}^{j<i}$). This helps, as the model sees imperfect context during training, but it still doesn't match the inference scenario where previous frames are clean but generated by the model itself.
    3.  **Self Forcing (SF) (This Work):** The model denoises the current frame ($x^i$) conditioned on its *own previously generated, clean* frames ($\hat{x}^{<i}$). This perfectly aligns the training process with the inference process, directly tackling exposure bias.

        As illustrated in **Figure 1**, TF and DF create a mismatch because the joint probability of the generated sequence does not equal the product of the conditional probabilities used during training. SF resolves this by ensuring the conditioning context comes from the model's own distribution.

        ![Figure 1: Training paradigms for AR video diffusion models. (a) In Teacher Forcing, the model is trained to denoise each frame conditioned on the preceding clean, ground-truth context frames. (b) In…](images/1.jpg)
        *该图像是图1，对比了自回归视频扩散模型的训练范式。(a)教师强制和(b)扩散强制使用真实上下文训练，导致曝光偏差及训练与推理分布不匹配，如公式 $$p(\hat{x}^1)p(\hat{x}^2|x^1)p(\hat{x}^3|x^1,x^2) \neq p(\hat{x}^1,\hat{x}^2,\hat{x}^3)$$ 所示。(c)本文自强制在训练时进行自回归生成并作为上下文，弥合了分布鸿沟，实现训练与推理的分布一致性，如公式 $$p(\hat{x}^1)p(\hat{x}^2|\hat{x}^1)p(\hat{x}^3|\hat{x}^1,\hat{x}^2) = p(\hat{x}^1,\hat{x}^2,\hat{x}^3)$$ 所示。*

    The paper also specifically calls out `CausVid` as a closely related work. `CausVid` used DMD loss but applied it to outputs generated via Diffusion Forcing. The authors of this paper argue this is a "critical flaw" because `CausVid` was matching the wrong distribution (the DF output distribution) instead of the true inference-time distribution.

# 4. Methodology (Core Technology & Implementation)

The core of the paper is the `Self Forcing` training algorithm, which is designed to be both effective and efficient.

*   **Principles:** The central idea is to make the training loop identical to the inference loop. By generating a full video sequence autoregressively and then applying a loss to the entire sequence, the model learns to handle and correct its own errors, becoming robust to the error accumulation that plagues TF/DF models.

*   **Steps & Procedures:**

    **1. Autoregressive Diffusion Post-Training via Self-Rollout (Section 3.2):**
    The training process for a single video is sequential, as shown in **Algorithm 1** and visualized in **Figure 2(c)**.

    ![Figure 2: Attention mask configurations. Both Teacher Forcing (a) and Diffusion Forcing (b) train the model on the entire video in parallel, enforcing causal dependencies with custom attention masks.…](images/2.jpg)
    *该图像是 Figure 2，展示了注意力掩码配置。其中，(a) Teacher Forcing 和 (b) Diffusion Forcing 通过自定义注意力掩码并行训练模型，以强制实现因果依赖。与此不同，(c) Self-Forcing Training 模仿自回归（AR）推理过程并利用 KV 缓存，不依赖特殊注意力掩码。图示场景中，视频包含3帧，每帧由2个token组成。*

    For each video frame $i$ from 1 to $N$:
    *   A full, few-step diffusion process is performed to generate the frame. It starts with Gaussian noise $x_{t_T}^i$.
    *   It is iteratively denoised for $T$ steps. At each step $j$, the model $G_{\theta}$ takes the noisy frame $x_{t_j}^i$ and the KV cache of previously *self-generated* clean frames ($\hat{x}^{<i}$) as input to predict a cleaner frame.
    *   The KV cache is updated with the final clean frame $\hat{x}^i$ before moving on to generate frame $i+1$.

    **2. Efficiency Optimizations:**
    A naive implementation of this would be computationally infeasible. The authors introduce two key optimizations:
    *   **Few-Step Diffusion:** Instead of hundreds of denoising steps per frame, they use a small number (e.g., 4 steps). This is a form of distillation where the model learns to take larger denoising leaps.
    *   **Stochastic Gradient Truncation:** Backpropagating through all frames and all denoising steps would consume too much memory. Instead, for each training sample, they randomly choose a single denoising step $s$ for each frame. The gradient is only calculated and backpropagated through this one step. This provides a stochastic but unbiased gradient signal to all parts of the denoising chain over the course of training. They also detach the gradients from the KV cache, so the gradient for the current frame does not flow back into previous frames.

    **Algorithm 1: Self Forcing Training**
    This pseudocode outlines the process. For each training loop:
    1.  Initialize an empty video output $Xθ$ and an empty KV cache `KV`.
    2.  Randomly sample a final denoising step $s$ for this iteration.
    3.  For each frame $i$ in the video:
        *   Start with noise $x_T^i$.
        *   Denoise from step $T$ down to $s$. For steps before $s$, gradients are disabled.
        *   At step $s$, enable gradients, compute the final clean frame $x_0$, and add it to the output video $Xθ$.
        *   Disable gradients again, compute the KV embeddings for $x_0$, and add them to the cache `KV`.
    4.  Once the full video $Xθ$ is generated, compute the holistic distribution matching loss and update the model parameters $\theta$.

*   **Holistic Distribution Matching Loss (Section 3.3):**
    Because `Self Forcing` generates a complete video sample from the model's true distribution $p_{\theta}(x^{1:N})$, it's possible to apply a loss that compares this distribution to the real data distribution $p_{data}(x^{1:N})$. The paper explores three such losses:
    *   **Distribution Matching Distillation (DMD):** Uses a pre-trained score model to estimate the difference between the generated and real data distributions and guides the generator to close this gap. It minimizes the reverse Kullback-Leibler (KL) divergence.
    *   **Score Identity Distillation (SiD):** A similar idea that minimizes the Fisher divergence between the two distributions.
    *   **Generative Adversarial Networks (GANs):** A video discriminator is trained to distinguish between the generated videos and real videos. The generator is trained to fool the discriminator.

        This is fundamentally different from TF/DF, which only minimize a per-frame loss conditioned on ground-truth context, effectively optimizing $\mathbb{E}_{x^{<i} \sim p_{data}} [D_{KL}(p_{data}(x^i|x^{<i}) || p_{\theta}(x^i|x^{<i}))]$.

*   **Long Video Generation with Rolling KV Cache (Section 3.4):**
    To generate videos longer than the training length, a sliding window approach is needed.
    *   **Problem:** Naive sliding windows are inefficient. Bidirectional models can't use KV caching and have $O(TL^2)$ complexity. Prior causal models recompute overlapping parts of the KV cache, leading to $O(L^2 + TL)$ complexity. (See **Figure 3**).
    *   **Solution:** The proposed `rolling KV cache` maintains a cache of a fixed size $L$. When generating a new frame that would exceed the cache size, the KV embedding of the oldest frame is discarded. This achieves a highly efficient $O(TL)$ complexity.
    *   **Artifact Mitigation:** A naive rolling cache can cause artifacts because the model was always trained seeing the very first frame's latent, which has different statistical properties. The solution is a simple but effective training trick: when training on a sequence, the attention mask for the final chunk is restricted so it *cannot* see the first chunk, simulating the condition it will encounter during long-form generation.

        ![Figure 3: Efficiency comparisons for video extrapolation. When performing video extrapolation through sliding window inference, (a) bidirectional diffusion models trained with TF/DF \[10, 73\] do not s…](images/3.jpg)

        **Algorithm 2: Inference with Rolling KV Cache**
    This pseudocode describes the generation process:
    1.  Initialize empty output $Xθ$ and KV cache `KV`.
    2.  For each frame $i$ to be generated:
        *   Perform the full $T$-step denoising process conditioned on the current `KV` cache.
        *   Append the final clean frame to $Xθ$.
        *   Compute its KV embedding.
        *   If the `KV` cache is full (size $L$), pop the oldest entry.
        *   Append the new KV embedding to the cache.
    3.  Return the generated video $Xθ$.

# 5. Experimental Setup

*   **Datasets:**
    *   **Training Prompts:** A filtered and LLM-extended version of the **VidProM** dataset, resulting in ~250k high-quality prompts.
    *   **GAN Training Data:** For the GAN-based objective, 70k videos were generated from a larger 14B parameter base model to serve as the "real" data distribution. This is a data-free distillation setup.
    *   **Evaluation Prompts:** **MovieGenBench** (1003 prompts) was used for the user study. **VBench** prompts were used for automated evaluation, also rewritten with an LLM for higher quality.

*   **Evaluation Metrics:**
    *   **VBench:** A comprehensive benchmark suite for evaluating generative video models across 16 different dimensions, including visual quality (e.g., `aesthetic quality`, `imaging quality`), temporal consistency (e.g., `motion smoothness`, `temporal flickering`), and semantic alignment (e.g., `object class`, `human action`). It provides a holistic view of a model's capabilities.
    *   **User Preference Study:** Human evaluators were shown two videos generated from the same prompt by different models and asked to choose which one was better overall, considering both quality and prompt alignment.
    *   **Throughput (FPS):** Frames Per Second. It measures how many video frames the model can generate per second. A value higher than the video's playback rate (e.g., 16 FPS) is needed for real-time performance.
        *   **Formula:** $FPS = \frac{\text{Total Frames Generated}}{\text{Total Time Taken}}$
    *   **Latency (s):** The time in seconds from when the generation request is made until the first frame is available for viewing. Low latency is critical for interactive applications.

*   **Baselines:** The paper compares `Self Forcing` against a strong set of open-source models:
    *   **Diffusion Models:** `Wan2.1-1.3B` (the base model they start from) and `LTX-Video`.
    *   **Chunk-wise AR Models:** `SkyReels-V2`, `MAGI-1`, and `CausVid`.
    *   **Frame-wise AR Models:** `NOVA` and `Pyramid Flow`.

# 6. Results & Analysis

*   **Core Results:**

    The main results are summarized in **Table 1** (transcribed below). The `Self Forcing` (chunk-wise) model achieves the highest VBench Total Score (84.31) among all compared models, even slightly surpassing its much slower, non-causal parent model `Wan2.1` (84.26). Critically, it does so with a throughput of 17.0 FPS and a latency of only 0.69 seconds, making it truly real-time. The frame-wise variant achieves even lower latency (0.45s) at a small cost to VBench score, making it suitable for highly interactive applications.

    *(Manual transcription of Table 1)*
    **Table 1: Comparison with relevant baselines.**

    <table>
    <thead>
    <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">#Params</th>
    <th rowspan="2">Resolution</th>
    <th rowspan="2">Throughput (FPS) ↑</th>
    <th rowspan="2">Latency (s) ↓</th>
    <th colspan="3">Evaluation scores ↑</th>
    </tr>
    <tr>
    <th>Total Score</th>
    <th>Quality Score</th>
    <th>Semantic Score</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="8"><strong>Diffusion models</strong></td>
    </tr>
    <tr>
    <td>LTX-Video [24]</td>
    <td>1.9B</td>
    <td>768×512</td>
    <td>8.98</td>
    <td>13.5</td>
    <td>80.00</td>
    <td>82.30</td>
    <td>70.79</td>
    </tr>
    <tr>
    <td>Wan2.1 [83]</td>
    <td>1.3B</td>
    <td>832×480</td>
    <td>0.78</td>
    <td>103</td>
    <td>84.26</td>
    <td>85.30</td>
    <td>80.09</td>
    </tr>
    <tr>
    <td colspan="8"><strong>Chunk-wise autoregressive models</strong></td>
    </tr>
    <tr>
    <td>SkyReels-V2 [10]</td>
    <td>1.3B</td>
    <td>960×540</td>
    <td>0.49</td>
    <td>112</td>
    <td>82.67</td>
    <td>84.70</td>
    <td>74.53</td>
    </tr>
    <tr>
    <td>MAGI-1 [69]</td>
    <td>4.5B</td>
    <td>832×480</td>
    <td>0.19</td>
    <td>282</td>
    <td>79.18</td>
    <td>82.04</td>
    <td>67.74</td>
    </tr>
    <tr>
    <td>CausVid [100]*</td>
    <td>1.3B</td>
    <td>832×480</td>
    <td>17.0</td>
    <td>0.69</td>
    <td>81.20</td>
    <td>84.05</td>
    <td>69.80</td>
    </tr>
    <tr>
    <td><strong>Self Forcing (Ours, chunk-wise)</strong></td>
    <td>1.3B</td>
    <td>832×480</td>
    <td>17.0</td>
    <td>0.69</td>
    <td><strong>84.31</strong></td>
    <td><strong>85.07</strong></td>
    <td><strong>81.28</strong></td>
    </tr>
    <tr>
    <td colspan="8"><strong>Autoregressive models†</strong></td>
    </tr>
    <tr>
    <td>NOVA [13]</td>
    <td>0.6B</td>
    <td>768×480</td>
    <td>0.88</td>
    <td>4.1</td>
    <td>80.12</td>
    <td>80.39</td>
    <td>79.05</td>
    </tr>
    <tr>
    <td>Pyramid Flow [33]</td>
    <td>2B</td>
    <td>640×384</td>
    <td>6.7</td>
    <td>2.5</td>
    <td>81.72</td>
    <td>84.74</td>
    <td>69.62</td>
    </tr>
    <tr>
    <td><strong>Self Forcing (Ours, frame-wise)</strong></td>
    <td>1.3B</td>
    <td>832×480</td>
    <td>8.9</td>
    <td><strong>0.45</strong></td>
    <td>84.26</td>
    <td>85.25</td>
    <td>80.30</td>
    </tr>
    </tbody>
    </table>

    The user study in **Figure 4** confirms these quantitative results, showing that human evaluators consistently prefer `Self Forcing` over all baselines, including its parent model `Wan2.1`.

    ![Figure 4: User preference study. Self Forcing outperforms all baselines in human preference.](images/4.jpg)
    *该图像是图4的用户偏好研究柱状图，展示了Self Forcing模型在视频生成方面优于所有对比基线的表现。其用户偏好率在54.2%到66.1%之间，显著超过了CausVid、Wan2.1、SkyReels-V2和MAGI-1模型。*

    Qualitative comparisons in **Figure 5** show that `Self Forcing` produces videos with high fidelity and temporal consistency, avoiding the color saturation artifacts that affect `CausVid`, which is a visual symptom of error accumulation.

    ![Figure 5: Qualitative comparisons. We visualize videos generated by Self Forcing (Ours) against those by Wan2.1 \[83\], SkyReels-V2 \[10\], and CausVid \[100\] at three time steps. All models share the sam…](images/5.jpg)
    *该图像是图5，展示了Self Forcing (Ours)与Wan2.1、SkyReels-V2和CausVid三种视频扩散模型在三个时间步（t=0s, t=2.5s, t=5s）上的定性比较结果。通过市场、玩耍的狗、冲浪的水獭及蜥蜴等多个场景的可视化视频序列，该图直观对比了不同模型生成的视频质量和时间连贯性。所有模型均采用1.3B参数的相同架构。*

*   **Ablations / Parameter Sensitivity:**

    **Table 2** (transcribed below) provides a controlled comparison of training paradigms. It shows that `Self Forcing` consistently outperforms `Teacher Forcing` and `Diffusion Forcing`, regardless of whether a many-step or few-step model is used. A key finding is that TF and DF performance degrades significantly when moving from chunk-wise to frame-wise generation (which involves more AR steps), while `Self Forcing` maintains its high quality. This directly demonstrates its success in mitigating error accumulation. The results also show that SF works well with all three distribution matching objectives (DMD, SiD, GAN).

    *(Manual transcription of Table 2)*
    **Table 2: Ablation study.**

    <table>
    <tr>
    <td>
    <table>

            <caption><strong>Chunk-wise AR</strong></caption>
            <thead>
              <tr>
                <th></th>
                <th colspan="3">Evaluation scores ↑</th>
              </tr>
              <tr>
                <th></th>
                <th>Total Score</th>
                <th>Quality Score</th>
                <th>Semantic Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colspan="4"><em>Many (50x2)-step models</em></td>
              </tr>
              <tr>
                <td>Diffusion Forcing (DF)</td>
                <td>82.95</td>
                <td>83.66</td>
                <td>80.09</td>
              </tr>
              <tr>
                <td>Teacher Forcing (TF)</td>
                <td>83.58</td>
                <td>84.34</td>
                <td>80.52</td>
              </tr>
              <tr>
                <td colspan="4"><em>Few (4)-step models</em></td>
              </tr>
              <tr>
                <td>DF + DMD</td>
                <td>82.76</td>
                <td>83.49</td>
                <td>79.85</td>
              </tr>
              <tr>
                <td>TF + DMD</td>
                <td>82.32</td>
                <td>82.73</td>
                <td>80.67</td>
              </tr>
              <tr>
                <td><strong>Self Forcing (Ours, DMD)</strong></td>
                <td><strong>84.31</strong></td>
                <td><strong>85.07</strong></td>
                <td><strong>81.28</strong></td>
              </tr>
              <tr>
                <td>Self Forcing (Ours, SiD)</td>
                <td>84.07</td>
                <td>85.52</td>
                <td>78.24</td>
              </tr>
              <tr>
                <td>Self Forcing (Ours, GAN)</td>
                <td>83.88</td>
                <td>85.06</td>
                <td>-</td>
              </tr>
            </tbody>
          </table>
        </td>
        <td>

        <table>

            <caption><strong>Frame-wise AR</strong></caption>
            <thead>
              <tr>
                <th></th>
                <th colspan="3">Evaluation scores ↑</th>
              </tr>
              <tr>
                <th></th>
                <th>Total Score</th>
                <th>Quality Score</th>
                <th>Semantic Score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colspan="4"><em>Many (50x2)-step models</em></td>
              </tr>
              <tr>
                <td>Diffusion Forcing (DF)</td>
                <td>77.24</td>
                <td>79.72</td>
                <td>67.33</td>
              </tr>
              <tr>
                <td>Teacher Forcing (TF)</td>
                <td>80.34</td>
                <td>81.34</td>
                <td>76.34</td>
              </tr>
              <tr>
                <td colspan="4"><em>Few (4)-step models</em></td>
              </tr>
              <tr>
                <td>DF + DMD</td>
                <td>80.56</td>
                <td>81.02</td>
                <td>78.71</td>
              </tr>
              <tr>
                <td>TF + DMD</td>
                <td>78.12</td>
                <td>79.62</td>
                <td>72.11</td>
              </tr>
              <tr>
                <td><strong>Self Forcing (Ours, DMD)</strong></td>
                <td><strong>84.26</strong></td>
                <td><strong>85.25</strong></td>
                <td><strong>80.30</strong></td>
              </tr>
              <tr>
                <td>Self Forcing (Ours, SiD)</td>
                <td>83.54</td>
                <td>-</td>
                <td>78.86</td>
              </tr>
              <tr>
                <td>Self Forcing (Ours, GAN)</td>
                <td>83.27</td>
                <td>-</td>
                <td>-</td>
              </tr>
            </tbody>
          </table>
        </td>
      </tr>
    </table>

*   **Rolling KV Cache Analysis:** **Figure 7** qualitatively shows the importance of the proposed training trick for the rolling KV cache. The naive implementation leads to severe blurring and artifacts in extrapolated frames, while the proposed method with a restricted attention window during training maintains high quality.

    ![Figure 7: Qualitative comparisons on video extrapolation. We present a visual comparison between the naive baseline and our proposed technique for rolling KV cache-based video extrapolation. Compared…](images/7.jpg)
    *该图像是图7，展示了滚动KV缓存视频外推的定性比较。它对比了使用朴素基线和我们提出的方法进行视频帧外推的效果。图像中，朴素基线外推的视频帧普遍存在严重的视觉伪影，如模糊和失真，尤其在细节处。相比之下，我们提出的方法能够生成更清晰、更逼真且没有明显伪影的视频帧，从而突显了其在维持视频质量方面的显著优势，特别是在处理视频外推任务时。*

*   **Training Efficiency:** **Figure 6** reveals a surprising and crucial finding. The per-iteration training time of `Self Forcing` is comparable to, or even faster than, parallel methods like TF and DF. This is because SF uses standard, highly optimized full attention kernels (like FlashAttention), whereas TF/DF require custom causal attention masks that are less optimized. The right panel shows that SF not only trains quickly but also converges to a higher quality level in the same amount of wall-clock time, making it superior in both performance and efficiency.

    ![Figure 6: Training efficiency comparison. Left: Per-iteration time across different chunk-wise, few-step autoregressive video diffusion training algorithms (using DMD as the distribution matching obj…](images/6.jpg)
    *该图像是图表，图6，展示了不同视频扩散模型训练效率的比较。左侧的柱状图对比了在生成器和判别器更新中，不同chunk-wise、few-step自回归视频扩散算法（包括Diffusion Forcing、Teacher Forcing及不同步数的Self Forcing）的每次迭代训练时间。右侧的折线图则展示了视频质量（VBench分数）随训练墙钟时间的变化。结果表明，Self Forcing通常具有更低的单次迭代训练时间，并在相同训练时间内达到更高的视频质量。*

# 7. Conclusion & Reflections

*   **Conclusion Summary:** The paper successfully introduces `Self Forcing`, a training paradigm that resolves the long-standing issue of exposure bias in autoregressive video models. By aligning the training and inference processes and optimizing a holistic, video-level loss, the method significantly reduces error accumulation. This allows for the creation of video generation models that are not only high-quality—matching or surpassing slower, non-causal systems—but also highly efficient, enabling real-time generation with low latency.

*   **Limitations & Future Work:** The authors acknowledge two main limitations:
    1.  Quality can still degrade when generating videos that are substantially longer than the context length seen during training.
    2.  The gradient truncation strategy, while necessary for efficiency, might limit the model's ability to learn very long-range temporal dependencies.
        Future research could focus on better extrapolation techniques and exploring alternative architectures like state-space models (`SSMs`) that are inherently recurrent and may better balance efficiency and long-context modeling.

*   **Personal Insights & Critique:**
    *   This paper presents a very elegant and powerful solution to a fundamental problem. The core idea of "train as you test" is not new, but its application to autoregressive diffusion models, combined with clever efficiency optimizations, is a significant contribution.
    *   The finding on training efficiency is particularly impactful. It challenges the conventional wisdom that parallelizable training is always superior and suggests that for certain sequence modeling tasks, a well-designed sequential training loop can be more efficient by leveraging optimized kernels.
    *   The proposed paradigm of **"parallel pre-training and sequential post-training"** is a compelling direction for the future. A large model can be pre-trained efficiently in a parallel fashion (like a standard DiT), and then fine-tuned with `Self Forcing` to adapt it for autoregressive inference. This combines the strengths of both worlds.
    *   The work beautifully integrates three major families of generative models: the sequential structure of AR models, the high-fidelity generation of Diffusion models, and the powerful distribution-matching principle from GANs. This synthesis demonstrates the complementary nature of these approaches.
    *   The authors also responsibly include a discussion on the **Broader Societal Impact** in the appendix, acknowledging the dual-use nature of real-time video generation technology and the potential for misuse in creating deepfakes and spreading disinformation. This is a crucial consideration for such powerful technology.