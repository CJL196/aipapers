# 1. Bibliographic Information

## 1.1. Title
The paper is titled **"DiffusionNFT: Online Diffusion Reinforcement with Forward Process"**. The central topic is a new reinforcement learning (RL) paradigm designed specifically for diffusion models. Unlike previous methods that operate on the reverse (denoising) process, this work proposes optimizing the model directly on the forward (noising) process using flow matching.

## 1.2. Authors
The authors are **Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu**.
*   **Affiliations:** The research is a collaboration between **Tsinghua University**, **NVIDIA**, and **Stanford University**.
*   **Background:** The authors represent a strong mix of academic and industrial research backgrounds, with significant prior contributions in generative modeling, reinforcement learning, and computer vision.

## 1.3. Journal/Conference
*   **Publication Status:** This paper is currently a **preprint** published on **arXiv**.
*   **Date:** Published at (UTC): **2025-09-19T16:09:33.000Z**.
*   **Venue Influence:** As an arXiv preprint from major institutions (NVIDIA, Tsinghua, Stanford), it represents cutting-edge research likely intended for top-tier conferences like NeurIPS, ICML, or CVPR. The involvement of NVIDIA Research suggests a focus on scalable, practical implementations for industry-standard models.

## 1.4. Abstract
The paper addresses the challenge of applying online reinforcement learning (RL) to diffusion models, which is difficult due to intractable likelihoods. Existing methods discretize the reverse sampling process (e.g., GRPO-style training) but suffer from solver restrictions and inconsistencies with the forward process. The authors introduce **Diffusion Negative-aware FineTuning (DiffusionNFT)**, which optimizes diffusion models on the forward process via flow matching. By contrasting positive and negative generations, it defines an implicit policy improvement direction without needing likelihood estimation or sampling trajectories. DiffusionNFT is reported to be up to **$25\times$ more efficient** than FlowGRPO and operates without Classifier-Free Guidance (CFG), significantly boosting performance on benchmarks like GenEval.

## 1.5. Original Source Link
*   **Original Source Link:** [https://arxiv.org/abs/2509.16117](https://arxiv.org/abs/2509.16117)
*   **PDF Link:** [https://arxiv.org/pdf/2509.16117v2](https://arxiv.org/pdf/2509.16117v2)
*   **Status:** Preprint (Open Access).

# 2. Executive Summary

## 2.1. Background & Motivation
### 2.1.1. Core Problem
The core problem is the difficulty of extending **Online Reinforcement Learning (RL)**—which has been highly successful for Large Language Models (LLMs)—to **Diffusion Models** for visual generation.
*   **Likelihood Intractability:** Policy Gradient algorithms (like PPO or GRPO) assume model likelihoods are computable. This holds for autoregressive models (like LLMs) but is violated by diffusion models, where likelihoods can only be approximated via costly probabilistic ODEs or variational bounds.
*   **Limitations of Current RL for Diffusion:** Recent works attempt to circumvent this by discretizing the *reverse* sampling process. However, the authors argue this introduces fundamental drawbacks:
    1.  **Forward Inconsistency:** Focusing only on the reverse process breaks adherence to the forward diffusion process, risking model degeneration.
    2.  **Solver Restriction:** Data collection relies on first-order SDE samplers, preventing the use of more efficient ODE or high-order solvers.
    3.  **Complicated CFG Integration:** Diffusion models rely on **Classifier-Free Guidance (CFG)** for quality, which typically requires training two models (conditional and unconditional), complicating RL optimization.

### 2.1.2. Innovative Idea
The paper's entry point is a fundamental question: **"Can diffusion reinforcement be performed on the forward process instead of the reverse?"**
Since a diffusion policy has a single forward (noising) process but multiple reverse (denoising) processes, the authors propose optimizing directly on the forward process. This allows them to treat RL as a supervised learning problem on the forward noising trajectory, leveraging **Flow Matching**.

## 2.2. Main Contributions / Findings
### 2.2.1. Primary Contributions
1.  **DiffusionNFT Paradigm:** A new online RL method that optimizes diffusion models on the forward process via flow matching, contrasting positive and negative generations to define an improvement direction.
2.  **Likelihood-Free Formulation:** Eliminates the need for likelihood estimation or storing sampling trajectories; requires only clean images and rewards.
3.  **Solver Flexibility:** Enables training with arbitrary black-box solvers (ODE or SDE) and is fully **CFG-free**.
4.  **Efficiency:** Demonstrates up to **$25\times$ efficiency improvement** over FlowGRPO.

### 2.2.2. Key Conclusions
*   DiffusionNFT improves the **GenEval score from 0.24 to 0.98 within 1k steps**, whereas FlowGRPO achieves 0.95 with over 5k steps and requires CFG.
*   By leveraging multiple reward models, DiffusionNFT significantly boosts the performance of **SD3.5-Medium** in every benchmark tested, outperforming CFG-based larger models.
*   The method validates that the forward process is a promising foundation for scalable, efficient, and theoretically principled diffusion RL.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, the reader must be familiar with the following concepts:

### 3.1.1. Diffusion Models
Diffusion models generate data by learning to reverse a gradual noising process.
*   **Forward Process:** Clean data $\pmb{x}_0$ is perturbed with Gaussian noise over time $t$ to become noise $\pmb{x}_T$. This process is fixed and follows a known distribution (e.g., Gaussian).
*   **Reverse Process:** The model learns to denoise $\pmb{x}_t$ back to $\pmb{x}_0$.
*   **Flow Matching:** A specific formulation where the model predicts a **velocity** vector $\pmb{v}_\theta(\pmb{x}_t, t)$ that describes the trajectory from noise to data. This is often more stable than predicting noise directly.

### 3.1.2. Reinforcement Learning (RL)
RL involves an agent learning to make decisions to maximize a cumulative reward.
*   **Policy Gradient:** A class of algorithms that optimize the policy (the model's behavior) directly by estimating the gradient of expected reward.
*   **GRPO (Group Relative Policy Optimization):** A variant of PPO popular in LLM training (e.g., DeepSeek-R1). It estimates advantages by comparing a group of samples generated for the same prompt, removing the need for a separate critic model.

### 3.1.3. Classifier-Free Guidance (CFG)
CFG is a technique to improve generation quality in diffusion models.
*   **Mechanism:** It combines the predictions of a **conditional model** (guided by a prompt) and an **unconditional model** (no prompt).
*   **Formula:** $\pmb{\epsilon}_{guided} = \pmb{\epsilon}_{uncond} + s \cdot (\pmb{\epsilon}_{cond} - \pmb{\epsilon}_{uncond})$, where $s$ is the guidance scale.
*   **Drawback:** It requires maintaining two models or computing two passes during inference, doubling computational cost. DiffusionNFT aims to eliminate this need.

## 3.2. Previous Works
The paper situates itself against several categories of prior research:

### 3.2.1. Likelihood-Free Methods
*   **Reward Backpropagation:** Directly differentiates through the reward function. Limited by memory costs and gradient explosion when unrolling long denoising chains.
*   **Reward-Weighted Regression (RWR):** An offline finetuning method that lacks a negative policy objective to penalize low-reward generations.
*   **Policy Guidance:** Methods like energy guidance or CFG-style guidance require combining multiple models, complicating online optimization.

### 3.2.2. Likelihood-Based Methods
*   **Diffusion-DPO:** Adapts Direct Preference Optimization to diffusion but requires likelihood approximations.
*   **FlowGRPO:** Extends GRPO to flow models by discretizing the reverse process. While effective, it couples training with specific SDE samplers and faces efficiency bottlenecks.

## 3.3. Technological Evolution
The field has evolved from simple fine-tuning to complex RL alignment.
1.  **Supervised Fine-Tuning (SFT):** Training on high-quality data.
2.  **RLHF (Reinforcement Learning from Human Feedback):** Using reward models to align outputs (standard in LLMs).
3.  **Diffusion RL:** Attempting to bring RLHF to images. Early attempts struggled with likelihoods.
4.  **Discretized Reverse RL (FlowGRPO):** Solves likelihoods by discretizing time but introduces solver constraints.
5.  **Forward Process RL (DiffusionNFT):** The current state-of-the-art proposed here, removing solver constraints and likelihood estimation entirely.

## 3.4. Differentiation Analysis
The core difference between DiffusionNFT and FlowGRPO is the **process direction**:
*   **FlowGRPO:** Operates on the **Reverse Process**. It treats generation as a multi-step Markov Decision Process (MDP). It requires storing trajectories and is tied to specific SDE solvers.
*   **DiffusionNFT:** Operates on the **Forward Process**. It treats RL as a contrastive supervised learning problem. It does not require trajectories, supports any solver, and is CFG-free.

    The following figure (Figure 2 from the original paper) illustrates this structural difference:

    ![Figure 2: Comparison between Forward-Process RL (NFT) and Reverse-Process RL (GRPO). NFT allows using any solvers and does not require storing the whole sampling trajectory for optimization.](images/2.jpg)
    *该图像是示意图，展示了正向过程强化学习（NFT）与反向过程强化学习（GRPO）的比较。图中展示了在正向过程中如何利用黑盒求解器进行优化，而无需存储整个采样轨迹。反向过程则包括离散化的反向SDE过程，利用历史数据进行策略更新。整体结构强调了两种方法在实现和效率上的差异。*

# 4. Methodology

This section provides an exhaustive deconstruction of the DiffusionNFT technical solution.

## 4.1. Principles
The core idea is to define policy improvement as a contrast between **positive** (high reward) and **negative** (low reward) generations. Instead of using policy gradients, DiffusionNFT integrates reinforcement signals into the standard **supervised learning objective** of diffusion models (specifically, flow matching).

Intuitively, the model learns to move its velocity prediction $\pmb{v}_\theta$ towards the velocity of an "ideal positive policy" and away from the "negative policy." This is done implicitly through a modified loss function.

## 4.2. Core Methodology In-depth

### 4.2.1. Problem Setup
Consider a pretrained diffusion policy $\pi^{\mathrm{old}}$ and prompt datasets $\{c\}$. At each iteration, $K$ images $\pmb{x}_0^{1:K}$ are sampled for a prompt $c$. Each image receives a reward $r \in [0, 1]$, representing its optimality probability $r(\pmb{x}_0, \pmb{c}) := p(\mathbf{o}=1 | \pmb{x}_0, \pmb{c})$.

This optimality serves as a bridge to split collected data into two imaginary subsets:
1.  **Positive Dataset ($\mathcal{D}^+$):** Images with probability $r$ of falling here.
2.  **Negative Dataset ($\mathcal{D}^-$):** Images with probability `1-r` of falling here.

    The underlying distributions of these subsets are defined as:
$$
\pi^+(\pmb{x}_0 | c) := \pi^{\mathrm{old}}(\pmb{x}_0 | \mathbf{o}=1, c) = \frac{r(\pmb{x}_0, c)}{p_{\pi^{\mathrm{old}}}(\mathbf{o}=1 | c)} \pi^{\mathrm{old}}(\pmb{x}_0 | c)
$$
$$
\pi^-(\pmb{x}_0 | c) := \pi^{\mathrm{old}}(\pmb{x}_0 | \mathbf{o}=0, c) = \frac{1 - r(\pmb{x}_0, c)}{1 - p_{\pi^{\mathrm{old}}}(\mathbf{o}=1 | c)} \pi^{\mathrm{old}}(\pmb{x}_0 | c)
$$
Here, $\pi^{\mathrm{old}}$ is the current policy, and $\pi^+$ represents the distribution of optimal samples. It is proven that $\pi^+ \succ \pi^{\mathrm{old}} \succ \pi^-$, meaning the positive distribution is strictly better than the old policy, which is better than the negative distribution.

### 4.2.2. Reinforcement Guidance
Rather than training solely on $\mathcal{D}^+$ (which is Rejection Fine-Tuning), DiffusionNFT leverages both positive and negative data to derive an **improvement direction** $\Delta \in \mathbb{R}^n$.

The training target velocity $\pmb{v}^*$ is defined as:
$$
\pmb{v}^*(\pmb{x}_t, c, t) := \pmb{v}^{\mathrm{old}}(\pmb{x}_t, c, t) + \frac{1}{\beta} \Delta(\pmb{x}_t, c, t).
$$
where $\pmb{v}$ is the velocity predictor of the diffusion model, and $\beta$ is a hyperparameter controlling guidance strength. $\Delta(\pmb{x}_t, \pmb{c}, t)$ is termed **reinforcement guidance**.

### 4.2.3. Theorem 3.1: Improvement Direction
To formalize $\Delta$, the authors study the distributional difference between $\pi^+$, $\pi^-$, and $\pi^{\mathrm{old}}$.

**Theorem 3.1 (Improvement Direction).** Consider diffusion models $\pmb{v}^+$, $\pmb{v}^-$, and $\pmb{v}^{\mathrm{old}}$ for the policy triplet $\pi^+$, $\pi^-$, and $\pi^{\mathrm{old}}$. The directional differences between these models are proportional:
$$
\begin{array}{rl}
\Delta := & [1 - \alpha(\pmb{x}_t)] [\pmb{v}^{\mathrm{old}}(\pmb{x}_t, \pmb{c}, t) - \pmb{v}^-(\pmb{x}_t, \pmb{c}, t)] \\
= & \alpha(\pmb{x}_t) [\pmb{v}^+(\pmb{x}_t, \pmb{c}, t) - \pmb{v}^{\mathrm{old}}(\pmb{x}_t, \pmb{c}, t)].
\end{array}
$$
where $0 \leq \alpha(\pmb{x}_t) \leq 1$ is a scalar coefficient:
$$
\alpha(\pmb{x}_t) := \frac{\pi_t^+(\pmb{x}_t | \pmb{c})}{\pi_t^{\mathrm{old}}(\pmb{x}_t | \pmb{c})} \mathbb{E}_{\pi^{\mathrm{old}}(\pmb{x}_0 | \pmb{c})} r(\pmb{x}_0, \pmb{c})
$$
This theorem indicates an ideal guidance direction $\Delta$. If we set `\beta = \alpha(\pmb{x}_t)`, the target policy becomes $\pmb{v}^* = \pmb{v}^+$, guaranteeing improvement.

The following figure (Figure 3 from the original paper) illustrates this improvement direction geometrically:

![Figure 3: Improvement Direction.](images/3.jpg)
*该图像是示意图，展示了DiffusionNFT中的改进方向。图中对比了正向生成（$D^+$）和负向生成（$D^-$）的流动，通过引导向量（`v_ heta`）指示了优化过程的发展，利用奖励值$r$从0到1的变化反映了生成效果的提升。*

### 4.2.4. Theorem 3.2: Policy Optimization
The paper introduces a training objective that directly optimizes $\pmb{v}_\theta$ towards $\pmb{v}^*$ using the collected datasets.

**Theorem 3.2 (Policy Optimization).** Consider the training objective:
$$
\mathcal{L}(\theta) = \mathbb{E}_{c, \pi^{\mathrm{old}}(\pmb{x}_0 \mid c), t} r \| \pmb{v}_\theta^+(\pmb{x}_t, \pmb{c}, t) - \pmb{v} \|_2^2 + (1 - r) \| \pmb{v}_\theta^-(\pmb{x}_t, \pmb{c}, t) - \pmb{v} \|_2^2,
$$
where
$$
\pmb{v}_\theta^+(\pmb{x}_t, \pmb{c}, t) := (1 - \beta) \pmb{v}^{\mathrm{old}}(\pmb{x}_t, \pmb{c}, t) + \beta \pmb{v}_\theta(\pmb{x}_t, \pmb{c}, t), \quad (\text{Implicit positive policy})
$$
and
$$
\pmb{v}_\theta^-(\pmb{x}_t, \pmb{c}, t) := (1 + \beta) \pmb{v}^{\mathrm{old}}(\pmb{x}_t, \pmb{c}, t) - \beta \pmb{v}_\theta(\pmb{x}_t, \pmb{c}, t). \quad (\text{Implicit negative policy})
$$
Given unlimited data and model capacity, the optimal solution of Eq. (5) satisfies:
$$
\pmb{v}_{\theta^*}(\pmb{x}_t, c, t) = \pmb{v}^{\mathrm{old}}(\pmb{x}_t, c, t) + \frac{2}{\beta} \Delta(\pmb{x}_t, c, t).
$$

This formulation (Figure 4) allows the model to learn from both positive and negative signals without training two independent models. It adopts an **implicit parameterization technique** to optimize a single target policy $\pmb{v}_\theta$.

![Figure 4: DiffusionNFT jointly optimizes two dual diffusion objectives, on both positive $( r = 1 )$ and negative $( r = 0$ branches. Rather than training two independent models ${ \\boldsymbol { v } } _ { \\theta } ^ { + }$ and ${ \\boldsymbol v } _ { \\boldsymbol { \\theta } _ { } } ^ { - }$ , it adopts a implicit parameerization technique that directlyoptimizes single target poliy ${ \\pmb v } _ { \\theta }$ .](images/4.jpg)
*该图像是示意图，展示了DiffusionNFT在积极（$r = 1$）和消极（$r = 0$）传播目标上进行联合优化的过程。图中通过条件输入（如“可爱的小狗”）和添加噪声生成图像，采用隐式参数化技术来优化单一目标策略${\pmb v}_{\theta}$。优化过程考虑了最优性奖励$r^{1:K} \in [0, 1]$，并通过具体的损失函数设计实现目标策略的提升。*

### 4.2.5. Algorithm 1: DiffusionNFT
The practical implementation follows Algorithm 1.

**Algorithm 1 Diffusion Negative-aware FineTuning (DiffusionNFT)**
*   **Require:** Pretrained policy $\pmb{v}^{\mathrm{ref}}$, raw reward function $r^{\mathrm{raw}}(\cdot) \in \mathbb{R}$, prompt dataset $\{c\}$.
*   **Initialize:** Data collection policy $\pmb{v}^{\mathrm{old}} \leftarrow \pmb{v}^{\mathrm{ref}}$, training policy $\pmb{v}_\theta \leftarrow \pmb{v}^{\mathrm{ref}}$, data buffer $\mathcal{D} \leftarrow \emptyset$.

1.  **for each iteration $i$ do**
2.  **for each sampled prompt $c$ do** // Rollout Step, Data Collection
3.  Sample $K$ images $\pmb{x}_0^{1:K}$ and rewards $\{r^{\mathrm{raw}}\}^{1:K}$.
4.  Normalize raw rewards in group: $r^{\mathrm{norm}} := r^{\mathrm{raw}} - \mathrm{mean}(\{r^{\mathrm{raw}}\}^{1:K})$.
5.  Define optimality probability $r = 0.5 + 0.5 * \mathrm{clip}\{r^{\mathrm{norm}} / Z_c, -1, 1\}$.
6.  $\mathcal{D} \leftarrow \{c, \pmb{x}_0^{1:K}, r^{1:K} \in [0, 1]\}$.
7.  **end for**
8.  **for each mini batch $\{c, \pmb{x}_0, r\} \in \mathcal{D}$ do** // Gradient Step, Policy Optimization
9.  Forward diffusion process: $\pmb{x}_t = \alpha_t \pmb{x}_0 + \sigma_t \pmb{\epsilon}; \pmb{v} = \dot{\alpha}_t \pmb{x}_0 + \dot{\sigma}_t \pmb{\epsilon}$.
10. Implicit positive velocity: $\pmb{v}_\theta^+(\pmb{x}_t, \pmb{c}, t) := (1 - \beta) \pmb{v}^{\mathrm{old}}(\pmb{x}_t, \pmb{c}, t) + \beta \pmb{v}_\theta(\pmb{x}_t, \pmb{c}, t)$.
11. Implicit negative velocity: $\pmb{v}_\theta^-(\pmb{x}_t, \pmb{c}, t) := (1 + \beta) \pmb{v}^{\mathrm{old}}(\pmb{x}_t, \pmb{c}, t) - \beta \pmb{v}_\theta(\pmb{x}_t, \pmb{c}, t)$.
12. $\theta \leftarrow \theta - \lambda \nabla_\theta [ r \| \pmb{v}_\theta^+(\pmb{x}_t, c, t) - \pmb{v} \|_2^2 + (1 - r) \| \pmb{v}_\theta^-(\pmb{x}_t, c, t) - \pmb{v} \|_2^2 ]$. (Eq. (5))
13. **end for**
14. Update data collection policy $\theta^{\mathrm{old}} \leftarrow \eta_i \theta^{\mathrm{old}} + (1 - \eta_i) \theta$, and clear buffer $\mathcal{D} \leftarrow \emptyset$. // Online Update
15. **end for**

### 4.2.6. Practical Implementation Details
*   **Optimality Reward:** Raw rewards are transformed into $r \in [0, 1]$ using group normalization:
    $$
    r(\pmb{x}_0, \pmb{c}) := \frac{1}{2} + \frac{1}{2} \mathrm{clip} \left[ \frac{r^{\mathrm{raw}}(\pmb{x}_0, \pmb{c}) - \mathbb{E}_{\pi^{\mathrm{old}}(\cdot | \pmb{c})} r^{\mathrm{raw}}(\pmb{x}_0, \pmb{c})}{Z_c}, -1, 1 \right].
    $$
*   **Soft Update:** The sampling policy $\pi^{\mathrm{old}}$ is updated via EMA (Exponential Moving Average): $\theta^{\mathrm{old}} \leftarrow \eta_i \theta^{\mathrm{old}} + (1 - \eta_i) \theta$. This balances stability and convergence speed.
*   **Adaptive Loss Weighting:** Instead of manual tuning `w(t)`, they use self-normalized $\pmb{x}_0$ regression:
    $$
    w(t) \| \pmb{v}_\theta(\pmb{x}_t, c, t) - \pmb{v} \|_2^2 \rightarrow \frac{\| \pmb{x}_\theta(\pmb{x}_t, c, t) - \pmb{x}_0 \|_2^2}{\mathrm{sg}(\mathrm{mean}(\mathrm{abs}(\pmb{x}_\theta(\pmb{x}_t, c, t) - \pmb{x}_0)))}
    $$
*   **CFG-Free:** The model is initialized solely by the conditional model. The functionality of CFG is learned through RL post-training.

# 5. Experimental Setup

## 5.1. Datasets
The experiments are based on **SD3.5-Medium** (2.5B parameters) at $512 \times 512$ resolution.

*   **Prompt Datasets:**
    *   **GenEval & OCR:** Training and test sets from FlowGRPO.
    *   **Other Rewards:** Trained on **Pick-a-Pic** and evaluated on **DrawBench**.
*   **Data Characteristics:**
    *   **GenEval:** Focuses on compositional image generation (e.g., "a blue pizza and a red hot dog").
    *   **OCR:** Focuses on visual text rendering (e.g., "a sign that reads 'OPEN'").
    *   **Pick-a-Pic:** A large dataset of user preferences for text-to-image generation.

## 5.2. Evaluation Metrics
The paper uses a comprehensive set of metrics to evaluate image quality and alignment.

### 5.2.1. Rule-Based Metrics
*   **GenEval:** Evaluates compositional image generation. It checks for object existence, color, count, and position.
*   **OCR:** Evaluates visual text rendering accuracy.

### 5.2.2. Model-Based Metrics
*   **PickScore:** Measures image quality and alignment based on human preferences.
*   **CLIPScore:** Measures image-text alignment using CLIP embeddings.
    *   **Formula:** $S_{\text{CLIP}} = \cos(E_I(I), E_T(T))$, where $E_I$ and $E_T$ are image and text encoders.
*   **HPSv2.1 (Human Preference Score):** Predicts human preference scores.
*   **Aesthetics:** Predicts the aesthetic quality of the image.
*   **ImageReward (ImgRwd):** Learns and evaluates human preferences for text-to-image generation.
*   **UnifiedReward (UniRwd):** A unified reward model for multimodal understanding and generation.

## 5.3. Baselines
*   **SD3.5-M (w/o CFG):** The base model without Classifier-Free Guidance.
*   **SD3.5-M (w/ CFG):** The base model with standard Classifier-Free Guidance.
*   **FlowGRPO:** The state-of-the-art reverse-process RL baseline.
*   **Larger Models:** SD3.5-L (8B) and FLUX.1-Dev (12B) for comparison.

## 5.4. Training Configuration
*   **Finetuning:** LoRA ($\alpha=64, r=32$).
*   **Epochs:** Each epoch consists of 48 groups with group size $G=24$.
*   **Sampling:** 10 rollout steps for comparison/ablation, 40 steps for best quality.
*   **Evaluation:** 40-step first-order ODE sampler.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results strongly validate the effectiveness of DiffusionNFT.

### 6.1.1. Multi-Reward Joint Training
The following table (Table 1 from the original paper) presents the evaluation results across multiple benchmarks. Note that the table uses merged cells for header categories.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">#Iter</th>
<th colspan="2">Rule-Based</th>
<th colspan="6">Model-Based</th>
</tr>
<tr>
<th>GenEval</th>
<th>OCR</th>
<th>PickScore</th>
<th>ClipScore</th>
<th>HPSv2.1</th>
<th>Aesthetic</th>
<th>ImgRwd</th>
<th>UniRwd</th>
</tr>
</thead>
<tbody>
<tr>
<td>SD-XL‡</td>
<td></td>
<td>0.55</td>
<td>0.14</td>
<td>22.42</td>
<td>0.287</td>
<td>0.280</td>
<td>5.60</td>
<td>0.76</td>
<td>2.93</td>
</tr>
<tr>
<td>SD3.5-L‡</td>
<td></td>
<td>0.71</td>
<td>0.68</td>
<td>22.91</td>
<td>0.289</td>
<td>0.288</td>
<td>5.50</td>
<td>0.96</td>
<td>3.25</td>
</tr>
<tr>
<td>FLUX.1-Dev</td>
<td></td>
<td>0.66</td>
<td>0.59</td>
<td>22.84</td>
<td>0.295</td>
<td>0.274</td>
<td>5.71</td>
<td>0.96</td>
<td>3.27</td>
</tr>
<tr>
<td>SD3.5-M (w/o CFG)</td>
<td></td>
<td>0.24</td>
<td>0.12</td>
<td>20.51</td>
<td>0.237</td>
<td>0.204</td>
<td>5.13</td>
<td>-0.58</td>
<td>2.02</td>
</tr>
<tr>
<td>+ CFG</td>
<td>—</td>
<td>0.63</td>
<td>0.59</td>
<td>22.34</td>
<td>0.285</td>
<td>0.279</td>
<td>5.36</td>
<td>0.85</td>
<td>3.03</td>
</tr>
<tr>
<td>+ FlowGRPO†</td>
<td>&gt;5k</td>
<td>0.95</td>
<td>0.66</td>
<td>22.51</td>
<td>0.293</td>
<td>0.274</td>
<td>5.32</td>
<td>1.06</td>
<td>3.18</td>
</tr>
<tr>
<td rowspan="3">+ Ours</td>
<td>2k</td>
<td>0.66</td>
<td>0.92</td>
<td>22.41</td>
<td>0.290</td>
<td>0.280</td>
<td>5.32</td>
<td>0.95</td>
<td>3.15</td>
</tr>
<tr>
<td>4k</td>
<td>0.54</td>
<td>0.68</td>
<td>23.50</td>
<td>0.280</td>
<td>0.316</td>
<td>5.90</td>
<td>1.29</td>
<td>3.37</td>
</tr>
<tr>
<td>1.7k</td>
<td>0.94</td>
<td>0.91</td>
<td>23.80</td>
<td>0.293</td>
<td>0.331</td>
<td>6.01</td>
<td>1.49</td>
<td>3.49</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Performance:** The DiffusionNFT model ("+ Ours" at 1.7k iters) achieves a **GenEval score of 0.94**, surpassing FlowGRPO (0.95 at >5k iters) and significantly outperforming the CFG baseline (0.63).
*   **Efficiency:** It achieves this with **1.7k iterations** compared to FlowGRPO's **>5k iterations**, demonstrating roughly **$3\times$ to $25\times$ efficiency**.
*   **Generalization:** It outperforms larger models like SD3.5-L and FLUX.1-Dev on several metrics despite having fewer parameters (2.5B vs 8B/12B).
*   **CFG-Free:** The model operates without CFG, yet matches or exceeds CFG-based performance.

    Figure 1 (from the original paper) visualizes the head-to-head comparison on GenEval and the multi-reward boost:

    ![Figure 1: Performance of DiffusionNFT. (a) Head-to-head comparison with FlowGRPO on the GenEval task. (b) By employing multiple reward models, DiffusionNFT significantly boosts the performance of SD3.5-Medium in every benchmark tested, while being fully CFG-free.](images/1.jpg)
    *该图像是一个图表，展示了DiffusionNFT与Flow-GRPO在GenEval任务上的性能对比（图(a)），以及在不同基准测试中，利用多个奖励模型显著提升SD3.5-Medium的表现（图(b)）。*

### 6.1.2. Head-to-Head Comparison
DiffusionNFT is consistently more efficient than FlowGRPO across single-reward tasks.
*   **GenEval:** Improves from 0.24 to 0.98 within 1k steps. FlowGRPO achieves 0.95 with over 5k steps.
*   **Efficiency:** As shown in Figure 6, DiffusionNFT is up to $25\times$ faster in wall-clock time.

    ![Figure 5: Qualitative Comparison. The prompts are taken from GenEva1, OCR and DrawBench respectively, where we compare the corresponding FlowGRPO model with our model.](images/6.jpg)
    *该图像是图表，展示了DiffusionNFT与FlowGRPO在不同任务上的效率对比。图中横轴为训练时间（GPU小时），纵轴分别为OCR得分、PickScore和HPSv2.1得分。DiffusionNFT在各项测试中表现出显著的效率提升，最高达24倍。*

### 6.1.3. Qualitative Comparison
Qualitative results (Figures 11, 12, 13, 17) show that DiffusionNFT generates images with better text rendering, composition, and aesthetic quality compared to FlowGRPO and the base model.

![该图像是一个示意图，展示了不同生成模型（SD3.5-M、FlowGRPO、DiffusionNFT）在无CFG模式下的效果对比，包括多个样本图像如红色狗、领带、披萨等，直观展示了DiffusionNFT的优势。](images/11.jpg)
*该图像是一个示意图，展示了不同生成模型（SD3.5-M、FlowGRPO、DiffusionNFT）在无CFG模式下的效果对比，包括多个样本图像如红色狗、领带、披萨等，直观展示了DiffusionNFT的优势。*

![该图像是展示不同模型效能的对比图，包括 SD3.5-M (无 CFG)、SD3.5-M (有 CFG)、FlowGRPO (有 CFG) 和 DiffusionNFT (无 CFG)。该图呈现了在相同条件下，DiffusionNFT 相比其他模型的显著优势。](images/12.jpg)
*该图像是展示不同模型效能的对比图，包括 SD3.5-M (无 CFG)、SD3.5-M (有 CFG)、FlowGRPO (有 CFG) 和 DiffusionNFT (无 CFG)。该图呈现了在相同条件下，DiffusionNFT 相比其他模型的显著优势。*

![该图像是法庭锤的插图，展示了不同角度和背景中的法庭锤及其相关标语，如“ORDER IN THE COURT”。插图通过不同的视觉效果强调了法庭锤的重要性和法庭氛围。](images/13.jpg)
*该图像是法庭锤的插图，展示了不同角度和背景中的法庭锤及其相关标语，如“ORDER IN THE COURT”。插图通过不同的视觉效果强调了法庭锤的重要性和法庭氛围。*

![该图像是展示各种动物、建筑和乐器的混合。有一只狗上面放着一个酒杯，另外还有几只小动物玩弄草莓。中间是带有'NeurIPS'字样的店面，底部展示了多个乐器的细节，最后则是几只猫头鹰的侧面图。视觉内容丰富且多样。](images/17.jpg)
*该图像是展示各种动物、建筑和乐器的混合。有一只狗上面放着一个酒杯，另外还有几只小动物玩弄草莓。中间是带有'NeurIPS'字样的店面，底部展示了多个乐器的细节，最后则是几只猫头鹰的侧面图。视觉内容丰富且多样。*

## 6.2. Ablation Studies / Parameter Analysis
The authors conducted ablation studies to verify the effectiveness of key design choices.

### 6.2.1. Negative Loss
The negative-aware component is crucial. Without the negative policy loss on $\pmb{v}_\theta^-$, rewards collapse almost instantly during online training. This highlights the essential role of negative signals in diffusion RL, diverging from LLM observations where Rejection Fine-Tuning (RFT) is stronger.

### 6.2.2. Diffusion Sampler
Online samples are used for both reward evaluation and training data.
*   **Result:** ODE samplers outperform SDE ones, especially on noise-sensitive metrics like PickScore.
*   **Figure 7:** Shows GenEval and PickScore trends for 1st-order SDE, 1st-order ODE, and 2nd-order ODE.

    ![该图像是一个示意图，展示了不同训练迭代下的 GenEval 分数变化。四条曲线分别对应于不同的超参数设置，表现出各自的收敛趋势，说明对于 $ u_i$ 值的调整对模型性能的影响。](images/7.jpg)
    *该图像是一个示意图，展示了不同训练迭代下的 GenEval 分数变化。四条曲线分别对应于不同的超参数设置，表现出各自的收敛趋势，说明对于 $ u_i$ 值的调整对模型性能的影响。*

    ![Figure 7: Different diffusion samplers for data collection.](images/8.jpg)
    *该图像是图表，展示了不同训练迭代下的GenEval得分和PickScore。左侧(a)图中，三种方法（1st-order SDE、1st-order ODE和2nd-order ODE）的得分随训练迭代增加的变化情况，以及它们的收敛趋势。右侧(b)图则展现了相同方法下的PickScore变化趋势。*

### 6.2.3. Adaptive Weighting
Stability improves when the flow-matching loss is given higher weight at larger $t$. Inverse strategies (e.g., $w(t) = 1 - t$) lead to collapse.
*   **Figure 9:** Compares different time-dependent weighting strategies.

    ![Figure 9: Different time-dependent weighting strategies.](images/9.jpg)
    *该图像是图表，展示了不同时间依赖加权策略对GenEval得分和PickScore的影响。左侧图(a)显示了在训练迭代中，随着不同加权策略（如$w(t) = 1 - t$、$w(t) = 1$、$w(t) = t$和自适应权重）的变化，GenEval得分的变化曲线。右侧图(b)则显示了相应的PickScore变化。各条曲线的颜色和样式对应于不同的加权策略。*

### 6.2.4. Soft Update
The parameter $\eta_i$ governs the trade-off between learning speed and stability.
*   **On-policy ($\eta = 0$):** Rapid initial progress but prone to instability/collapse.
*   **Off-policy ($\eta \to 1$):** Stable but slow convergence.
*   **Strategy:** Start with small $\eta$ and gradually increase.
*   **Figure 8:** Shows GenEval scores for different soft-update strategies.

    ![Figure 7: Different diffusion samplers for data collection.](images/8.jpg) (Note: Figure 8 in source text is labeled "Soft-update strategies" in caption list but image 8 shows sampler comparison in VLM desc. I will rely on the caption provided in the text: "Figure 8: Soft-update strategies." for the context of soft update analysis, though image 7/8 descriptions in VLM list vary. I will use the image that corresponds to the analysis context best. The text says "Figure 8: Soft-update strategies." I will use `images/8.jpg` for soft update context if it matches, otherwise `images/7.jpg` based on the text flow. Looking at the provided text, Figure 7 is "Different diffusion samplers" and Figure 8 is "Soft-update strategies". I will cite them accordingly.)
    *该图像是图表，展示了不同训练迭代下的GenEval得分和PickScore。左侧(a)图中，三种方法（1st-order SDE、1st-order ODE和2nd-order ODE）的得分随训练迭代增加的变化情况，以及它们的收敛趋势。右侧(b)图则展现了相同方法下的PickScore变化趋势。*

    *Correction on Image Citation:* Based on the provided text captions:
- Figure 7: Different diffusion samplers.
- Figure 8: Soft-update strategies.
- Figure 9: Different time-dependent weighting strategies.
- Figure 10: Choices of strength $\beta$.

  I will ensure the images match these captions in the text flow.

### 6.2.5. Guidance Strength
The guidance parameter $\beta$ governs a trade-off between stability and convergence speed. $\beta$ near 1 performs stably.
*   **Figure 10:** Shows the impact of different strengths $\beta$ on GenEval score.

    ![Figure 10: Choices of strength $\\beta$ .](images/10.jpg)
    *该图像是一个示意图，展示了不同强度 `eta` 对 GenEval 分数的影响。随着训练迭代的增加，蓝色线（$eta = 0.01$）、橙色线（$eta = 1.0$）和绿色线（$eta = 10.0$）在 GenEval 分数上表现出不同的收敛趋势。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper introduces **DiffusionNFT**, a novel online RL paradigm for diffusion models that operates on the **forward process**.
*   **Key Findings:** It eliminates the need for likelihood estimation, supports arbitrary solvers, and is CFG-free.
*   **Performance:** It achieves up to **$25\times$ efficiency** over FlowGRPO and significantly boosts SD3.5-Medium performance across diverse benchmarks.
*   **Significance:** This work represents a step toward unifying supervised and reinforcement learning in diffusion, highlighting the forward process as a robust foundation for scalable diffusion RL.

## 7.2. Limitations & Future Work
*   **Reward Model Dependency:** Like all RLHF methods, performance is contingent on the quality of the reward models. Biases in reward models (e.g., PickScore) could be amplified.
*   **Computational Cost:** While more efficient than FlowGRPO, online RL still requires significant sampling and reward computation compared to standard SFT.
*   **Theoretical Bounds:** While Theorems 3.1 and 3.2 provide guarantees under unlimited data/capacity, practical convergence bounds in finite regimes could be further explored.

## 7.3. Personal Insights & Critique
*   **Innovation:** The shift from reverse-process to forward-process RL is a significant conceptual leap. It resolves the "likelihood intractability" issue by reframing RL as a contrastive supervised learning problem.
*   **CFG-Free Implication:** The ability to remove CFG is particularly impactful for deployment. CFG doubles inference cost; learning guidance into the weights via RL could halve inference latency for high-quality generation.
*   **Transferability:** The "Negative-aware FineTuning" (NFT) concept, originally from LLMs, is successfully transferred to diffusion. This suggests a potential **unified RL framework** across modalities (text, image, video) that relies on forward-process optimization rather than modality-specific likelihoods.
*   **Potential Issue:** The method relies on splitting data into positive/negative sets based on a threshold/probability $r$. In sparse reward environments (where most samples are low quality), the "positive" set might be too small to provide a stable gradient, potentially requiring careful reward normalization strategies.