# 1. Bibliographic Information

## 1.1. Title
The title of the paper is **"AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization"**. This title indicates the paper's central focus: aligning **Autoregressive (AR)** video generation models using a method called **Contrastive Policy Optimization (CoPO)**. The goal is to improve the quality and alignment of generated videos with human preferences.

## 1.2. Authors
The authors of this paper are:
*   **Dailan He** (CUHK MMLab, Vivix Group Limited)
*   **Guanlin Feng** (Vivix Group Limited)
*   **Xingtong Ge** (Vivix Group Limited, HKUST)
*   **Yi Zhang** (Vivix Group Limited)
*   **Bingqi Ma** (Vivix Group Limited)
*   **Guanglu Song** (Vivix Group Limited)
*   **Yu Liu** (Vivix Group Limited)
*   **Hongsheng Li** (CUHK MMLab, Shenzhen Loop Area Institute, CPII under InnoHK)

    The affiliations suggest a collaboration between academic institutions (CUHK, HKUST) and industry research groups (Vivix Group, Shenzhen Loop Area Institute). This combination often facilitates access to large-scale computational resources and practical deployment scenarios.

## 1.3. Journal/Conference
The paper was published as a **preprint on arXiv**.
*   **Publication Date:** March 18, 2026 (UTC).
*   **Source Link:** [https://arxiv.org/abs/2603.17461](https://arxiv.org/abs/2603.17461)
*   **PDF Link:** [https://arxiv.org/pdf/2603.17461v1.pdf](https://arxiv.org/pdf/2603.17461v1.pdf)

    As an arXiv preprint, it has not yet undergone peer review for a specific conference or journal at the time of this analysis, but it represents cutting-edge research in the field of generative video models and reinforcement learning.

## 1.4. Publication Year
The paper is dated **2026**, indicating it is a very recent contribution to the field of AI video generation.

## 1.5. Abstract
The abstract summarizes the research objective, methodology, and key findings:
*   **Objective:** To align streaming **Autoregressive (AR)** video generators with human preferences using **Reinforcement Learning from Human Feedback (RLHF)**.
*   **Problem:** Existing methods based on **Stochastic Differential Equations (SDE)** and **Group Relative Policy Optimization (GRPO)** fail in this setting because few-step AR models are near-deterministic and sensitive to initial noise, making intermediate exploration ineffective.
*   **Methodology:** The authors propose **AR-CoPO**, which adapts the **Neighbor GRPO** contrastive perspective. It introduces **chunk-level alignment** via a forking mechanism and a **semi-on-policy training strategy** that uses a replay buffer.
*   **Results:** Experiments on the **Self-Forcing** model show improvements in out-of-domain generalization and in-domain human preference alignment without reward hacking.

## 1.6. Original Source Link
The official source is the arXiv repository. The status is **Preprint**.

# 2. Executive Summary

## 2.1. Background & Motivation
### 2.1.1. Core Problem
The core problem addressed is the difficulty of aligning **streaming autoregressive (AR) video generators** using **Reinforcement Learning from Human Feedback (RLHF)**. While diffusion and flow-matching models have achieved high-quality video synthesis, their inference cost scales linearly with video length, making them unsuitable for low-latency, streaming applications. To solve this, recent work distills these models into causal AR generators that operate chunk-by-chunk with few-step sampling. However, aligning these distilled models with human preferences remains challenging.

### 2.1.2. Challenges in Prior Research
Existing **GRPO (Group Relative Policy Optimization)** methods designed for flow-matching models typically convert deterministic **Ordinary Differential Equation (ODE)** sampling into a stochastic **SDE** process to enable exploration. This approach faces two major challenges in the few-step AR setting:
1.  **Model Deviation:** Few-step generators (often consistency models) deviate from standard flow-matching ODEs.
2.  **Ineffective Exploration:** Short, low-stochasticity trajectories are highly sensitive to initialization noise. Intermediate SDE noise injections (used in standard GRPO) have negligible impact on the output, rendering exploration ineffective.

    The following figure (Figure 2 from the original paper) illustrates this failure: SDE-based GRPO fails to improve rewards, and perturbing intermediate noise produces nearly identical outputs compared to perturbing initial noise.

    ![Fig. 2: Left: Training curves comparing SDE-based GRPO and AR-CoPO on Self-Forcing. SDE-based GRPO fails to improve the reward, while AR-CoPO consistently achieves higher scores throughout training. Right: Perturbing only the intermediate CM solver noise (Rows 35) produces nearly identical outputs, whereas replacing the initial noise (Row 2) causes significant variation, confirming that few-step AR models (e.g. Self-Forcing \[4\]) are near-deterministic and driven primarily by initial noise.](images/2.jpg)
    *该图像是图表与样例图的组合。左侧展示了SDE-based GRPO与AR-CoPO在Self-Forcing上的训练曲线对比，AR-CoPO的表现优于SDE-based GRPO。右侧显示不同步骤生成的视频样本，初始噪声的变化显著影响最终输出，显示了少步AR模型对初始噪声的敏感性。*

### 2.1.3. Innovative Entry Point
The paper's entry point is the **Neighbor GRPO** perspective, which reinterprets SDE-GRPO updates as a distance-driven contrastive objective over neighbor candidate trajectories. This suggests that exploration can be controlled during training by constructing neighborhoods around the **initial noise** rather than relying on intermediate SDE noise. The authors adapt this to the streaming AR setting by introducing **chunk-level alignment**.

## 2.2. Main Contributions / Findings
### 2.2.1. Primary Contributions
1.  **AR-CoPO Framework:** A novel framework adapting Neighbor GRPO for streaming AR video generation.
2.  **Chunk-Level Alignment:** A forking mechanism that constructs neighborhood candidates at a randomly selected chunk, enabling localized credit assignment.
3.  **Semi-On-Policy Training:** A strategy complementing on-policy exploration with exploitation over a replay buffer of reference rollouts to improve generation quality.
4.  **LoRA Merging:** A method to merge on-policy and semi-on-policy adapters to balance exploration and exploitation.

### 2.2.2. Key Findings
*   AR-CoPO consistently achieves higher reward scores compared to SDE-based GRPO baselines.
*   The method improves both **out-of-domain generalization** (measured by VBench) and **in-domain human preference alignment** (measured by VideoAlign).
*   The dual-benchmark criterion (improving both benchmarks) provides evidence of genuine alignment rather than reward hacking.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, readers need familiarity with several key concepts in generative AI and reinforcement learning.

### 3.1.1. Autoregressive (AR) Video Generation
**Autoregressive (AR)** models generate data sequentially, where each new part depends on the previously generated parts. In video generation, this means generating video frames or chunks one after another. This contrasts with **bidirectional generation** (common in diffusion models), where the entire sequence is generated simultaneously. AR generation is crucial for **streaming** applications where low latency is required.

### 3.1.2. Flow Matching and ODEs
**Flow Matching** is a generative modeling technique that learns a vector field to transport noise to data. Sampling is often done by solving an **Ordinary Differential Equation (ODE)**.
*   **ODE (Ordinary Differential Equation):** A deterministic equation describing how a system changes over time. In generative models, solving the ODE transforms random noise into a clean image/video.
*   **SDE (Stochastic Differential Equation):** Similar to an ODE but includes a random noise term. SDEs are stochastic, meaning they introduce randomness during sampling, which is useful for exploration in RL but slower for inference.

### 3.1.3. Reinforcement Learning from Human Feedback (RLHF)
**RLHF** is a technique to align AI models with human preferences. It involves training a **reward model** on human preference data and then using **Reinforcement Learning (RL)** to optimize the generative model to maximize this reward.
*   **Policy:** The generative model itself, viewed as a policy that takes actions (generating tokens/frames).
*   **Reward:** A score indicating how good the generated output is according to human preferences.

### 3.1.4. GRPO (Group Relative Policy Optimization)
**GRPO** is a variant of **Proximal Policy Optimization (PPO)** designed for generative models. Instead of training a separate critic model to estimate values, GRPO computes advantages by comparing the rewards of a group of samples generated from the same prompt. This reduces variance and computational cost.

### 3.1.5. LoRA (Low-Rank Adaptation)
**LoRA** is a parameter-efficient fine-tuning technique. Instead of updating all model weights, LoRA adds small, trainable low-rank matrices to the existing weights. This allows for efficient training and easy merging of different adapters (e.g., one for exploration, one for exploitation).

### 3.1.6. Consistency Models
**Consistency Models** are a type of generative model designed for fast sampling. They map noisy inputs directly to clean outputs in few steps (often 1-4 steps), bypassing the long iterative process of standard diffusion models. This makes them ideal for real-time applications but challenging for RL alignment due to their near-deterministic nature.

## 3.2. Previous Works
### 3.2.1. Neighbor GRPO
The paper builds upon **Neighbor GRPO** [2], which reinterprets SDE-based GRPO as a contrastive objective. Neighbor GRPO constructs neighbors by perturbing initial noise and defines a surrogate policy based on distances in latent space. This allows for inference-time determinism (ODE sampling) while enabling training-time exploration.

### 3.2.2. Self-Forcing and Causal-Forcing
**Self-Forcing** [4] and **Causal-Forcing** [33] are baseline AR video generation models. They distill bidirectional video models into causal AR generators. Self-Forcing is the primary baseline used in this paper's experiments. These models use few-step ODE solvers or consistency models for fast generation.

### 3.2.3. SDE-Based GRPO Variants
Methods like **Dance-GRPO** [26] and **FlowGRPO** [10] convert ODE sampling to SDE for RL alignment. The paper argues these are ineffective for few-step AR models because intermediate noise injections do not significantly alter the output.

## 3.3. Technological Evolution
The field has evolved from bidirectional diffusion models (high quality, high latency) to distilled flow-matching models (faster) to autoregressive models (streaming, low latency). Alignment techniques have evolved from supervised fine-tuning to RLHF (PPO, GRPO). This paper represents the convergence of **streaming AR generation** and **contrastive RL alignment**, addressing the specific incompatibility between SDE-based RL and few-step deterministic samplers.

## 3.4. Differentiation Analysis
Compared to standard SDE-GRPO, **AR-CoPO** differs in:
1.  **Exploration Source:** AR-CoPO perturbs **initial chunk noise** rather than intermediate SDE noise.
2.  **Granularity:** AR-CoPO operates at the **chunk level** (forking at a specific chunk) rather than the sequence level, reducing cost and improving credit assignment.
3.  **Training Strategy:** AR-CoPO introduces **semi-on-policy training** with a replay buffer, whereas standard GRPO is typically purely on-policy.

# 4. Methodology

## 4.1. Principles
The core principle of **AR-CoPO** is to align streaming AR video generators by treating the sampling process as a policy rollout and optimizing it using a contrastive objective derived from **Neighbor GRPO**. The key intuition is that since few-step AR models are near-deterministic and driven by initial noise, exploration should be controlled by perturbing the initial noise of specific chunks during training, while keeping inference deterministic.

## 4.2. Core Methodology In-depth
The AR-CoPO training pipeline consists of three phases: Rollout, Reward, and Replay & Update.

### 4.2.1. Chunk-Level Alignment via Forking
To handle the streaming nature of AR generation, AR-CoPO introduces **chunk-level alignment**. Instead of forking the entire sequence, the model forks at a randomly selected **pivot chunk**.

1.  **Shared Context Generation:** A pivot chunk index $p$ is sampled uniformly from $\{1, \ldots, L\}$, where $L$ is the sequence length in chunks. The model generates the first `p-1` chunks to establish a shared historical context $h_{p-1}$ (e.g., cached KV states).
2.  **Action Space Forking:** At the $p$-th chunk, the generation branches into $G$ neighbors. This is done by perturbing a shared base initial noise $\epsilon_p^*$. The perturbed noises are calculated as:
    \$
    \epsilon ^ { ( i ) } = \sqrt { 1 - \sigma ^ { 2 } } \epsilon ^ { * } + \sigma \delta ^ { ( i ) } , \quad \delta ^ { ( i ) } \sim \mathcal { N } ( 0 , I ) , \quad i = 1 , \ldots , G ,
    \$
    where $\sigma \in (0, 1)$ controls the exploration radius.
    For each branch $i$, the model completes the denoising generation to produce the chunk latent $x_p^{(i)}$.
3.  **Rollout and Sequence-Level Reward:** For each of the $G$ branches, the model deterministically generates the remaining `L-p` chunks. A sequence-level reward $r^{(i)}$ is computed for each completed branch.

    **Controlled Noise Sharing:** Crucially, within each iteration, the only randomness differing across branches is the initial noise of the pivot chunk $\epsilon_p^{(i)}$. All other noise sources (non-pivot chunks, CM solver noises) are shared. This ensures reward differences are attributed solely to the pivot chunk's generation.

The following figure (Figure 3 from the original paper) illustrates this training pipeline.

![Fig. 3: The AR-CoPO training pipeline. (1) Rollout: The model autoregressively generates a shared context up to a randomly selected pivot chunk $p$ At chunk $p$ , the base initial noise is perturbed into $G$ neighbors; each neighbor is forked into an independent branch and autoregressively completed to produce a full video sequence. (2) Reward: Each completed sequence is decoded and scored by a reward model, yielding a sequence-level reward per branch. (3) Replay $\\&$ Update: The saved pivotchunk trajectories are replayed through the current policy; distances between current and old $\\scriptstyle { \\hat { x } } _ { 0 }$ predictions induce surrogate policy ratios, which are used in a clipped GRPO update confined to the pivot chunk.](images/3.jpg)
*该图像是AR-CoPO训练流水线的示意图，展示了模型如何通过自回归生成共享上下文，利用随机扰动生成邻居，并通过评分模型计算序列奖励。图中分为三个部分：1. 回放：生成初始噪声和共享上下文，进行自回归采样；2. 奖励：对完成序列进行解码和评分；3. 回放与更新：通过GRPO更新策略，使用旧预测与新预测之间的距离来生成奖励。*

### 4.2.2. Contrastive Policy Optimization (CoPO) Update
After collecting rewards, the model performs a **GRPO update** confined to the pivot chunk.

First, advantages $A^{(i)}$ are computed using the sequence-level rewards:
\$
A ^ { ( i ) } = \frac { r ^ { ( i ) } - \bar { r } } { \sigma _ { \bar { r } } }
\$
where $\bar{r}$ is the mean reward and $\sigma_{\bar{r}}$ is the standard deviation of the rewards in the group.

Next, a surrogate training-time transition distribution $\pi_\theta(i)$ is defined based on the distance between the current policy's prediction and the candidate latents. For **Flow-Matching (FM)** based generators, distances are computed in intermediate latent space $x_t$. However, for **Consistency Models (CM)**, distances are defined in the $\hat{x}_0$ prediction space.

For Consistency Models, the distance $d_{0,t}^{(i)}$ and policy $\pi_\theta(i \mid s_t)$ are defined as:
\$
d _ { 0 , t } ^ { ( i ) } = \left\| \hat { x } _ { 0 , t } ^ { ( i ) } - \hat { x } _ { 0 , t } ^ { ( \theta ) } \right\| _ { 2 } ^ { 2 } , \qquad \pi _ { \theta } ( i \mid s _ { t } ) = \frac { \exp \Bigl ( - d _ { 0 , t } ^ { ( i ) } / \tau _ { 0 } \Bigr ) } { \sum _ { k = 1 } ^ { G } \exp \Bigl ( - d _ { 0 , t } ^ { ( k ) } / \tau _ { 0 } \Bigr ) } ,
\$
where $\hat{x}_{0,t}^{(i)} = F_{\theta_{old}}(x_t^{(i)}, h_{t-1}, t)$ is produced by the old parameters, $\hat{x}_{0,t}^{(\theta)}$ is the current policy's prediction, and $\tau_0$ is a temperature hyperparameter.

Finally, the model parameters $\theta$ are optimized to maximize the GRPO objective:
\$
J ( \theta ) = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } \left( \frac { \pi _ { \theta } ( i ) } { \pi _ { \mathrm { o l d } } ( i ) } A ^ { ( i ) } , \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( i ) } { \pi _ { \mathrm { o l d } } ( i ) } , 1 - \epsilon , 1 + \epsilon \right) A ^ { ( i ) } \right) .
\$
This objective pulls the anchor towards candidates with positive advantages and pushes it away from those with negative advantages. The update is restricted to the $p$-th chunk to save computation.

### 4.2.3. Semi-On-Policy Training Strategy
Pure on-policy exploration can be unstable for global semantic rewards like **Text Alignment (TA)**. To address this, AR-CoPO introduces a **semi-on-policy** strategy.

*   **On-Policy:** Rolls out fresh candidates from the evolving policy $\pi_\theta$ at each iteration (Exploration).
*   **Semi-On-Policy:** Fixes rollouts to a reference policy $\pi_{ref}$ (the initialization checkpoint) and uses a replay buffer of pre-collected candidates (Exploitation).

    The semi-on-policy objective applies the same contrastive GRPO update over fixed reference rollouts. **Ratio clipping** is retained to enforce a trust region and prevent distributional shift.

The following figure (Figure 4 from the original paper) contrasts these two training paradigms.

![Fig. 4: On-policy vs. semi-on-policy training under AR-CoPO. Left: On-policy training rolls out fresh candidates from the evolving policy $\\pi \\theta$ at each iteration, enabling active exploration of new generation modes guided by the reward signal. Right: Semion-policy training fixes all rollouts to a reference policy $\\pi _ { \\mathrm { r e f } }$ ; the contrastive objective upweights high-reward candidates and suppresses low-reward ones within a trust region maintained by ratio clipping, enhancing exploitation without sacrificing stability. Each paradigm trains an independent LoRA adapter; merging the two adapters yields the final aligned model that benefits from both exploration and exploitation.](images/4.jpg)
*该图像是示意图，展示了AR-CoPO中的两种训练方式：左侧为On-Policy训练，候选生成模式通过奖励信号进行主动探索；右侧为Semi-On-Policy训练，样本固定，增强了对高奖励候选的利用，双方各自训练独立的LoRA适配器。*

**LoRA Merging:** The on-policy and semi-on-policy objectives are optimized independently using separate **LoRA adapters**. At inference time, the adapters are merged by scaling the on-policy weights. This allows the model to benefit from both exploration (reward improvement) and exploitation (quality preservation).

### 4.2.4. Algorithm Summary
The complete training procedure for one iteration is summarized in Algorithm 1 below.

**Algorithm 1 AR-CoPO Training (one iteration)**
*   **Require:** Policy $\theta$, reward $r(\cdot)$, sequence length $L$, group size $G$
1.  Sample pivot $p \sim \mathrm{Uniform}(1, L)$
2.  Generate shared context $h_{p-1}$ by running $\theta$ on chunks $1, \ldots, p-1$
3.  **for** $i = 1, \dots, G$ **do** (Fork at chunk $p$)
4.      $\epsilon_p^{(i)} \gets \sqrt{1 - \sigma^2} \epsilon_p^* + \sigma \delta^{(i)}, \quad \delta^{(i)} \sim \mathcal{N}(0, I)$
5.      Denoise chunk $p$ from $\epsilon_p^{(i)}$; complete remaining chunks; compute $r^{(i)}$
6.  **end for**
7.  $A^{(i)} \gets (r^{(i)} - \bar{r}) / \sigma_r$
8.  Replay chunk $p$, compute $\pi_\theta(i) \propto \exp(-\|\hat{x}_0^{(i)} - \hat{x}_0^{(\theta)}\|^2 / \tau_0)$
9.  Update $\theta$ via GRPO (Eq. 3) on chunk $p$ only

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted on the **MovieGen Video Bench** [15].
*   **Source:** A benchmark dataset for evaluating video generation models.
*   **Characteristics:** It contains diverse text prompts for generating videos.
*   **Usage:** Used for training and evaluating the alignment of the video generators.
*   **Example Prompts:** The paper provides qualitative examples such as "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas" and "An astronaut flying in space, featuring a steady and smooth perspective."

## 5.2. Evaluation Metrics
The paper uses two main benchmark suites: **VBench** and **VideoAlign**.

### 5.2.1. VBench
**VBench** is a comprehensive benchmark for video generation.
*   **Quality:** Measures visual fidelity (e.g., resolution, lack of artifacts).
*   **Semantic:** Measures alignment with the text prompt (e.g., object presence, attributes).
*   **Total:** A weighted combination of Quality and Semantic scores.
*   **Formula:** While the paper does not explicitly provide the mathematical formula for VBench scores, they are typically computed as weighted averages of various dimension scores (e.g., $Score_{Total} = w_1 \cdot Score_{Quality} + w_2 \cdot Score_{Semantic}$).

### 5.2.2. VideoAlign
**VideoAlign** [11] is a reward suite specifically designed for RLHF alignment.
*   **VQ (Video Quality):** Rewards high visual quality.
*   **MQ (Motion Quality):** Rewards smooth and plausible motion.
*   **TA (Text Alignment):** Rewards faithfulness to the input text prompt.
*   **Overall:** A combination of VQ, MQ, and TA.
*   **Formula:** Similar to VBench, the Overall score is an aggregation of the component rewards. The paper treats these as reward signals $r^{(i)}$ during training.

## 5.3. Baselines
The paper compares AR-CoPO against several strong baselines:
1.  **Self-Forcing [4]:** The primary baseline, a few-step streaming AR video generator.
2.  **Causal-Forcing [33]:** Another AR distillation method.
3.  **LongLive [27]:** A representative few-step streaming AR video generator.
4.  **SDE-Based GRPO:** A baseline following the design of Dance-GRPO [26] and FlowGRPO [10], which converts ODE sampling to SDE for RL.

## 5.4. Implementation Details
*   **Fine-tuning:** All models are fine-tuned with **LoRA** (rank 64, $\alpha=128$).
*   **Hardware:** Training is conducted on 24 GPUs.
*   **Hyperparameters:** Group size $G=12$, learning rate $1 \times 10^{-5}$, initial noise perturbation strength $\sigma=0.5$.
*   **Iterations:** Models are evaluated after 100 training iterations.

# 6. Results & Analysis

## 6.1. Core Results Analysis
### 6.1.1. Quantitative Comparison
The main quantitative results are presented in Table 1. AR-CoPO demonstrates improvements over the baselines.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">VBench</th>
<th colspan="4">VideoAlign</th>
</tr>
<tr>
<th>Quality</th>
<th>Semantic</th>
<th>Total</th>
<th>VQ</th>
<th>MQ</th>
<th>TA</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td>Self-Forcing</td>
<td>84.87</td>
<td>71.27</td>
<td>82.15</td>
<td>3.80</td>
<td>1.68</td>
<td>2.28</td>
<td>7.76</td>
</tr>
<tr>
<td>Causal-Forcing</td>
<td>85.27</td>
<td>70.35</td>
<td>82.28</td>
<td>3.97</td>
<td>1.43</td>
<td>2.40</td>
<td>7.79</td>
</tr>
<tr>
<td>LongLive</td>
<td>85.10</td>
<td>71.16</td>
<td>82.31</td>
<td>3.87</td>
<td>1.76</td>
<td>2.43</td>
<td>8.06</td>
</tr>
<tr>
<td>Self-Forcing + ours (semi)</td>
<td>85.15</td>
<td>71.68</td>
<td>82.45</td>
<td>3.70</td>
<td>1.60</td>
<td>2.30</td>
<td>7.61</td>
</tr>
<tr>
<td>Self-Forcing + ours (on-policy)</td>
<td>84.81</td>
<td>70.71</td>
<td>81.99</td>
<td>4.15</td>
<td>2.06</td>
<td>2.30</td>
<td>8.51</td>
</tr>
<tr>
<td>Self-Forcing + ours (merged)</td>
<td>85.07</td>
<td>70.55</td>
<td>82.17</td>
<td>4.00</td>
<td>1.86</td>
<td>2.36</td>
<td>8.22</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Semi-On-Policy:** The semi-on-policy model ("+ ours (semi)") surpasses all streaming AR baselines on **VBench Total** (82.45 vs. 82.31 for LongLive), showing effective exploitation.
*   **Merged Model:** After merging the on-policy adapter ("+ ours (merged)"), **VideoAlign Overall** improves significantly from 7.76 to 8.22, while **VBench Total** is maintained (82.15→82.17). This confirms genuine alignment rather than reward hacking.
*   **On-Policy Alone:** The on-policy model achieves the highest VideoAlign Overall (8.51) but degrades VBench Total (81.99), indicating potential over-optimization or reward hacking.

### 6.1.2. Qualitative Comparison
Figure 5 shows side-by-side comparisons between AR-CoPO and Self-Forcing. AR-CoPO produces videos with better aesthetic quality, more vivid appearance, and better adherence to text descriptions.

![该图像是示意图，展示了三个不同的生成示例：第一部分是根据提示“鸟和猫”生成的鸟类与猫咪的图像；第二部分基于提示“A cute happy Corgi playing in park, sunset”生成了可爱的柯基犬的图像；第三部分则是根据相同提示生成的像素艺术风格的柯基犬图像。](images/5.jpg)
*该图像是示意图，展示了三个不同的生成示例：第一部分是根据提示“鸟和猫”生成的鸟类与猫咪的图像；第二部分基于提示“A cute happy Corgi playing in park, sunset”生成了可爱的柯基犬的图像；第三部分则是根据相同提示生成的像素艺术风格的柯基犬图像。*

## 6.2. Comparison with SDE-GRPO
The paper compares AR-CoPO against an SDE-based GRPO baseline.
*   **Training Curves:** As shown in Figure 2 (Left) and Figure 7, SDE-based GRPO fails to improve the reward, while AR-CoPO steadily achieves higher scores.
*   **Reason for Failure:** Few-step AR models are near-deterministic. Diversity is governed by initial noise, not intermediate solver noise. SDE-GRPO freezes initial noise and perturbs intermediate noise, resulting in near-zero policy gradient signals.

    The following figure (Figure 6 from the original paper) validates this by showing that replacing initial noise causes substantial variation, while replacing intermediate noise produces marginal changes.

    ![Fig. 6: Analysis of entropy sources in Self-Forcing. Each sub-figure corresponds to forking at a different chunk position. Row 1: Reference sample with all noise frozen. Row 2: Only the initial noise of the forked chunk is replaced—the output changes substantially. Rows 35: Only the CM solver noise at a specific denoising timestep within the chunk is replaced—the output changes marginally. This confirms that sample diversity in Self-Forcing is governed almost entirely by the initial noise, making intermediate SDE-style noise injection ineffective as an exploration mechanism.](images/7.jpg)
    *该图像是示意图，展示了在自我增强(Self-Forcing)中不同位置分叉的分析。每个子图展示了参考样本、初始噪声替换以及在各时间步下的变化，说明样本多样性主要受初始噪声的影响。图中包含的核心信息为（a）和（b）所示的分叉位置。*

## 6.3. Ablation Studies
### 6.3.1. Training Strategies
Table 2 ablates the training strategies (on-policy, off-policy, semi-on-policy) when optimizing only the **Text Alignment (TA)** reward.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">VBench</th>
<th colspan="4">VideoAlign</th>
</tr>
<tr>
<th>Quality</th>
<th>Semantic</th>
<th>Total</th>
<th>VQ</th>
<th>MQ</th>
<th>TA</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td>Self-Forcing</td>
<td>84.87</td>
<td>71.27</td>
<td>82.15</td>
<td>3.80</td>
<td>1.68</td>
<td>2.28</td>
<td>7.76</td>
</tr>
<tr>
<td>on-policy</td>
<td>81.66</td>
<td>69.68</td>
<td>79.26</td>
<td>3.53</td>
<td>0.25</td>
<td>2.63</td>
<td>6.42</td>
</tr>
<tr>
<td>off-policy</td>
<td>69.78</td>
<td>60.84</td>
<td>67.99</td>
<td>2.22</td>
<td>-0.15</td>
<td>2.16</td>
<td>4.23</td>
</tr>
<tr>
<td>semi-on-policy</td>
<td>85.15</td>
<td>71.68</td>
<td>82.45</td>
<td>3.70</td>
<td>1.60</td>
<td>2.30</td>
<td>7.61</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **On-Policy Collapse:** On-policy training improves TA (2.28→2.63) but causes severe degradation in **Motion Quality (MQ)** (1.68→0.25) and **VBench Total**. This is identified as **reward hacking**, where the model sacrifices temporal coherence to maximize semantic scores.
*   **Semi-On-Policy Stability:** Semi-on-policy training avoids this collapse, maintaining scores close to the baseline while improving VBench Quality and Semantic.
*   **Off-Policy Instability:** Fully off-policy training (no ratio clipping) causes drastic deterioration, confirming the necessity of the trust region.

    Figure 8 illustrates the temporal inconsistencies introduced by on-policy training compared to the stability of semi-on-policy training.

    ![该图像是插图，展示了一对情侣在户外雨天共享伞下的亲密瞬间。图中情侣表现出不同的情感互动与互动细节，体现了温馨的氛围和浪漫的场景。](images/8.jpg)
    *该图像是插图，展示了一对情侣在户外雨天共享伞下的亲密瞬间。图中情侣表现出不同的情感互动与互动细节，体现了温馨的氛围和浪漫的场景。*

### 6.3.2. LoRA Merging Scales
Table 3 analyzes the effect of merging the on-policy weights at different scales.

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Scale</th>
<th colspan="3">VBench</th>
<th colspan="4">VideoAlign</th>
</tr>
<tr>
<th>Quality</th>
<th>Semantic</th>
<th>Total</th>
<th>VQ</th>
<th>MQ</th>
<th>TA</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td>1.0</td>
<td>84.90</td>
<td>70.38</td>
<td>81.99</td>
<td>4.13</td>
<td>1.86</td>
<td>2.34</td>
<td>8.33</td>
</tr>
<tr>
<td>0.8</td>
<td>85.07</td>
<td>70.55</td>
<td>82.17</td>
<td>4.00</td>
<td>1.86</td>
<td>2.36</td>
<td>8.22</td>
</tr>
<tr>
<td>0.6</td>
<td>85.11</td>
<td>70.72</td>
<td>82.23</td>
<td>3.86</td>
<td>1.78</td>
<td>2.36</td>
<td>7.99</td>
</tr>
<tr>
<td>0.4</td>
<td>85.14</td>
<td>71.44</td>
<td>82.40</td>
<td>3.76</td>
<td>1.62</td>
<td>2.34</td>
<td>7.72</td>
</tr>
<tr>
<td>0 (Semi)</td>
<td>85.15</td>
<td>71.68</td>
<td>82.45</td>
<td>3.70</td>
<td>1.60</td>
<td>2.30</td>
<td>7.61</td>
</tr>
</tbody>
</table>

**Analysis:**
*   There is a clear trade-off: increasing the on-policy scale improves VideoAlign Overall but degrades VBench Total.
*   **Scale Selection:** The authors select **Scale = 0.8** as the default because it satisfies the **dual-improvement criterion** (improves VideoAlign Overall from 7.76 to 8.22 while maintaining VBench Total at 82.17). Scale 1.0 achieves higher VideoAlign scores but degrades VBench, indicating over-optimization.

## 6.4. Performance on Causal-Forcing
The paper also evaluates AR-CoPO on the **Causal-Forcing** baseline (Table 4 in Appendix). Results show consistent gains, with a LoRA merging scale of 0.5 achieving the best balance. This confirms the broad applicability of AR-CoPO across different AR backbones.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper presents **AR-CoPO**, a framework for aligning few-step streaming autoregressive video generators. By adapting the **Neighbor GRPO** contrastive perspective to the chunk-level structure of AR generation, AR-CoPO circumvents the mismatch between SDE-based exploration and near-deterministic consistency model samplers. The **semi-on-policy training strategy** further enhances quality by exploiting high-quality reference rollouts within a trust region. Experiments demonstrate that AR-CoPO improves both out-of-domain generalization (VBench) and in-domain human preference alignment (VideoAlign), providing evidence of genuine alignment.

## 7.2. Limitations & Future Work
### 7.2.1. Limitations
*   **Compute Cost:** While chunk-level forking reduces cost compared to sequence-level forking, generating $G$ branches for every pivot chunk still incurs significant computational overhead during training.
*   **Reward Hacking Risk:** The ablation study shows that pure on-policy optimization is prone to reward hacking (sacrificing motion quality for text alignment). This requires careful balancing via the semi-on-policy strategy and LoRA merging.
*   **Dependency on Reference Policy:** The semi-on-policy strategy relies on a high-quality reference policy. If the initialization model is poor, the exploitation phase may be limited.

### 7.2.2. Future Work
*   **Adaptive Forking:** Instead of random pivot selection, future work could explore adaptive strategies to fork at chunks with higher uncertainty or lower quality.
*   **Multi-Reward Balancing:** Further research could investigate dynamic weighting of different reward components (VQ, MQ, TA) during training to prevent collapse in specific dimensions.
*   **Extension to Longer Videos:** Evaluating the method on significantly longer video sequences to test scalability and long-term consistency.

## 7.3. Personal Insights & Critique
### 7.3.1. Strengths
*   **Problem Identification:** The paper sharply identifies the incompatibility between standard SDE-GRPO and few-step AR models. The analysis of entropy sources (Figure 6) is particularly convincing.
*   **Practical Solution:** The chunk-level forking and semi-on-policy strategy are practical engineering solutions that address the specific constraints of streaming generation.
*   **Rigorous Evaluation:** The use of a **dual-benchmark criterion** (VBench + VideoAlign) to detect reward hacking is a strong methodological contribution. It prevents the common pitfall of optimizing for a reward metric at the expense of general quality.

### 7.3.2. Potential Issues
*   **Hyperparameter Sensitivity:** The performance depends heavily on the LoRA merging scale (Table 3). Finding the optimal scale requires additional validation runs, which adds to the tuning cost.
*   **Generalization:** While tested on Self-Forcing and Causal-Forcing, it remains to be seen how well AR-CoPO generalizes to other AR architectures or non-video modalities (e.g., AR audio generation).

### 7.3.3. Transferability
The core idea of **chunk-level contrastive alignment** could be transferred to other sequential generation tasks, such as **long-form text generation** or **music generation**, where full-sequence RL is computationally prohibitive. The **semi-on-policy** concept of separating exploration and exploitation adapters is also a generalizable pattern for stabilizing RLHF training.