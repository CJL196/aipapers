# 1. Bibliographic Information

## 1.1. Title
Flow-GRPO: Training Flow Matching Models via Online RL

## 1.2. Authors
Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Lil, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang.

**Affiliations:**
The authors are affiliated with prominent academic institutions and technology companies:
*   MMLab, CUHK (The Chinese University of Hong Kong)
*   Tsinghua University
*   Kling Team, Kuaishou Technology
*   Nanjing University
*   Shanghai AI Laboratory

    Their backgrounds collectively suggest expertise in artificial intelligence, machine learning, computer vision, and deep learning, particularly in generative models and reinforcement learning.

## 1.3. Journal/Conference
The paper is published as a preprint on arXiv, specifically `arXiv:2505.05470`. While not yet peer-reviewed in a formal journal or conference proceedings, arXiv is a highly influential platform for rapid dissemination of research in AI and related fields, allowing researchers to share their work before or concurrently with formal publication processes.

## 1.4. Publication Year
2025

## 1.5. Abstract
This paper introduces Flow-GRPO, a novel method that integrates online policy gradient reinforcement learning (RL) into flow matching models for the first time. The approach relies on two key strategies: (1) an `ODE-to-SDE` conversion, which transforms the deterministic Ordinary Differential Equation (ODE) underlying flow matching into an equivalent Stochastic Differential Equation (SDE). This conversion preserves the model's marginal distribution across all timesteps and introduces the necessary stochasticity for `RL exploration`. (2) A `Denoising Reduction` strategy, which significantly reduces the number of denoising steps required during training while maintaining the original inference steps, thereby boosting sampling efficiency without compromising performance. Empirically, Flow-GRPO demonstrates strong effectiveness across various text-to-image tasks. For compositional generation, an `RL-tuned SD3.5-M` (Stable Diffusion 3.5 Medium) model achieves a near-perfect increase in GenEval accuracy from $63\%$ to $95\%$ for tasks involving object counts, spatial relations, and fine-grained attributes. In visual text rendering, accuracy improves from $59\%$ to $92\%$. The method also shows substantial gains in aligning with human preferences. Notably, the improvements are achieved with minimal `reward hacking`, meaning that increases in reward did not lead to appreciable degradation in image quality or diversity.

## 1.6. Original Source Link
*   **Original Source Link:** `https://arxiv.org/abs/2505.05470`
*   **PDF Link:** `https://arxiv.org/pdf/2505.05470v5.pdf`
    The paper is currently a preprint on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem that this paper aims to solve revolves around the limitations of current flow matching models in generating complex and precise images, particularly for tasks requiring compositional accuracy and text rendering. While flow matching models, like those used in advanced image generation (e.g., `SD3.5-M`), have strong theoretical foundations and produce high-quality images, they often struggle with:
1.  **Composing complex scenes:** This includes accurately rendering multiple objects, their attributes, and spatial relationships (e.g., "a red ball on a blue box").
2.  **Visual text rendering:** Generating accurate and coherent text within images.

    This problem is important because text-to-image (T2I) generation models are increasingly expected to handle sophisticated prompts that demand fine-grained control and reasoning, moving beyond merely producing aesthetically pleasing but semantically inconsistent images. The gap in prior research is that while `online reinforcement learning (RL)` has proven highly effective in enhancing the reasoning capabilities of `Large Language Models (LLMs)`, its potential for advancing flow matching generative models remains largely unexplored. Previous applications of RL to generative models have mainly focused on `early diffusion-based models` or `offline RL techniques` (like `Direct Preference Optimization (DPO)`) for flow-based models.

The paper's innovative idea is to leverage `online RL`, specifically the `Gradient Policy Optimization (GRPO)` algorithm, to fine-tune flow matching models. This introduces two critical challenges:
1.  **Deterministic Nature of Flow Models:** Flow models rely on deterministic `Ordinary Differential Equations (ODEs)` for generation, which conflicts with `RL's` need for stochastic sampling to explore the environment.
2.  **Sampling Efficiency:** `Online RL` requires efficient data collection, but flow models typically need many iterative steps to generate each sample, making `RL` training costly and slow, especially for large models.

## 2.2. Main Contributions / Findings
The paper's primary contributions are:
1.  **First Online RL for Flow Matching:** Proposing `Flow-GRPO`, the first method to successfully integrate `online policy gradient RL` (specifically `GRPO`) into flow matching models, demonstrating its effectiveness for `T2I` tasks. This addresses the challenge of extending `RL's` benefits from `LLMs` to `T2I` generation with flow models.
2.  **ODE-to-SDE Conversion:** Developing a novel `ODE-to-SDE` strategy that transforms the deterministic `ODE-based` flow into an equivalent `Stochastic Differential Equation (SDE)` framework. This crucial step introduces the necessary randomness for `RL exploration` while preserving the original model's marginal distributions, overcoming the fundamental conflict between deterministic generative processes and `RL's` stochastic requirements.
3.  **Denoising Reduction Strategy:** Introducing a practical `Denoising Reduction` strategy that significantly reduces the number of denoising steps during `RL` training (e.g., from 40 to 10 steps) while maintaining the original number of inference steps during testing. This dramatically improves sampling efficiency and accelerates the training process without sacrificing the quality of the final generated images.
4.  **Effective `KL` Constraint for Reward Hacking Prevention:** Demonstrating that the `Kullback-Leibler (KL)` constraint effectively prevents `reward hacking`, where models optimize for the reward metric at the expense of overall image quality or diversity. Properly tuned `KL` regularization allows matching high rewards while preserving image quality, albeit with longer training.
5.  **Empirical Validation and Significant Performance Gains:**
    *   **Compositional Generation:** `Flow-GRPO` improves `SD3.5-M` accuracy on the GenEval benchmark from $63\%$ to $95\%$, even surpassing `GPT-4o`.
    *   **Visual Text Rendering:** Accuracy increases from $59\%$ to $92\%$.
    *   **Human Preference Alignment:** Achieves substantial gains in aligning with human preferences (e.g., Pickscore).
    *   **Minimal Reward Hacking:** All improvements are achieved with very little degradation in image quality or diversity, as evidenced by stable `DrawBench` metrics.

        These findings collectively address the core problem by significantly enhancing the reasoning and control capabilities of flow matching models, making them more robust and aligned with complex user intentions, without compromising the high-fidelity image generation they are known for.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

To understand `Flow-GRPO`, a reader should be familiar with several core concepts in machine learning and generative models:

*   **Generative Models:** These are models that can learn the distribution of input data and then generate new samples that resemble the training data. Examples include `Generative Adversarial Networks (GANs)`, `Variational Autoencoders (VAEs)`, `Diffusion Models`, and `Flow Matching Models`.
*   **Flow Matching Models:**
    *   **Continuous-Time Normalizing Flows:** These models transform a simple probability distribution (e.g., Gaussian noise) into a complex data distribution (e.g., images) through a continuous, invertible transformation. This transformation is defined by an `Ordinary Differential Equation (ODE)`.
    *   **Velocity Field:** The `ODE` describes how a data point $x$ evolves over time $t$ (from noise $x_1$ to data $x_0$). The `velocity field` $\pmb{v}_t(\pmb{x}_t, t)$ dictates the direction and speed of this transformation at any given point $\pmb{x}_t$ and time $t$. Flow matching models are trained to directly regress this `velocity field`.
    *   **Deterministic Sampling:** In standard flow matching, once the `velocity field` is learned, generating a sample involves numerically solving the `ODE` from a noise sample $\pmb{x}_1$ (e.g., standard Gaussian) to a data sample $\pmb{x}_0$. This process is deterministic, meaning the same initial noise will always produce the same output image.
*   **Reinforcement Learning (RL):**
    *   **Agent, Environment, State, Action, Reward:** `RL` involves an `agent` interacting with an `environment`. The `environment` is characterized by `states` $s$. At each `state`, the `agent` chooses an `action` $a$. This `action` leads to a new `state` and the `agent` receives a `reward` $R$. The goal of `RL` is for the `agent` to learn a `policy` that maximizes the cumulative `reward` over time.
    *   **Policy:** A `policy` $\pi(a|s)$ is a function that maps states to a probability distribution over actions, indicating which action to take in a given state.
    *   **Online RL vs. Offline RL:**
        *   **Online RL:** The `agent` learns by directly interacting with the `environment` and collecting new data (trajectories) on the fly, updating its `policy` iteratively. This allows for exploration and adaptation.
        *   **Offline RL:** The `agent` learns from a fixed dataset of previously collected interactions, without further interaction with the `environment`. This can be more sample-efficient but limits exploration.
    *   **Policy Gradient Methods:** A class of `RL` algorithms that directly optimize the `policy` function (e.g., a neural network) by taking gradients of an objective function that represents the expected `reward`.
    *   **Exploration vs. Exploitation:** A fundamental dilemma in `RL`. `Exploration` involves trying new actions to discover better outcomes, while `exploitation` involves choosing actions known to yield high `rewards`. `Online RL` inherently relies on exploration.
*   **Markov Decision Process (MDP):** A mathematical framework for modeling sequential decision-making. An `MDP` is defined by a tuple $(S, \mathcal{A}, \rho_0, P, R)$:
    *   $S$: A set of possible `states`.
    *   $\mathcal{A}$: A set of possible `actions`.
    *   $\rho_0$: The initial `state` distribution.
    *   $P(s'|s,a)$: The `transition probability` function, defining the probability of reaching state $s'$ from state $s$ by taking action $a$.
    *   `R(s,a)`: The `reward` function, specifying the immediate reward received after taking action $a$ in state $s$.
*   **Ordinary Differential Equations (ODEs) and Stochastic Differential Equations (SDEs):**
    *   **ODE:** An equation involving an unknown function of one independent variable and its derivatives. `ODEs` describe deterministic continuous-time processes. In generative models, they describe the path from noise to data. For example, $\mathrm{d}\pmb{x}_t = \pmb{v}_t \mathrm{d}t$.
    *   **SDE:** An `ODE` extended with a `stochastic` (random) term, typically involving a Wiener process (or Brownian motion). `SDEs` describe continuous-time processes that are subject to random fluctuations. For example, $\mathrm{d}\pmb{x}_t = f(\pmb{x}_t, t)\mathrm{d}t + \sigma_t \mathrm{d}\pmb{w}$, where $\mathrm{d}\pmb{w}$ represents increments of a Wiener process and $\sigma_t$ is a diffusion coefficient controlling the noise level. The key difference is the introduction of `stochasticity`.
*   **Kullback-Leibler (KL) Divergence:** A measure of how one probability distribution $P$ diverges from a second, expected probability distribution $Q$. A low `KL divergence` means the two distributions are very similar. In `RL`, it's often used as a regularization term $D_{\mathrm{KL}}(\pi_{\theta} || \pi_{\mathrm{ref}})$ to keep the learned `policy` $\pi_{\theta}$ from deviating too much from a reference `policy` $\pi_{\mathrm{ref}}$, preventing aggressive policy updates that could lead to instability or `reward hacking`.
    *   Formula for two Gaussian distributions $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$:
        \$
        D_{\mathrm{KL}}(\mathcal{N}(\mu_1, \Sigma_1) || \mathcal{N}(\mu_2, \Sigma_2)) = \frac{1}{2} \left( \mathrm{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T \Sigma_2^{-1}(\mu_2-\mu_1) - k + \ln\left(\frac{\det(\Sigma_2)}{\det(\Sigma_1)}\right) \right)
        \$
        where $k$ is the dimensionality of the distributions. For isotropic Gaussians with $\Sigma_1 = \sigma_1^2 I$ and $\Sigma_2 = \sigma_2^2 I$:
        \$
        D_{\mathrm{KL}}(\mathcal{N}(\mu_1, \sigma_1^2 I) || \mathcal{N}(\mu_2, \sigma_2^2 I)) = \frac{1}{2} \left( \frac{\sigma_1^2}{\sigma_2^2} + \frac{||\mu_2-\mu_1||^2}{\sigma_2^2} - k + k \ln\left(\frac{\sigma_2^2}{\sigma_1^2}\right) \right)
        \$
*   **GRPO (Gradient Policy Optimization):** A `policy gradient` method mentioned as a lightweight alternative to `PPO` [20]. It's more memory-efficient by not requiring a `value network` and uses a group-relative advantage formulation.

## 3.2. Previous Works

The paper builds upon and differentiates itself from several lines of prior research:

*   **Flow Matching (FM) Models:**
    *   **Rectified Flow [3]:** A key foundational work that defines a straight path between data and noise, simplifying the `ODE` and enabling efficient deterministic sampling. This is the framework adopted by recent advanced models like `SD3.5-M` [4] and `FLUX.1 Dev` [5].
    *   **Flow Matching for Generative Modeling [2]:** Introduced the concept of learning `ODEs` by directly matching the `velocity field`, providing a solid theoretical basis.
*   **Diffusion Models:**
    *   **Denoising Diffusion Probabilistic Models (DDPM) [21]:** A seminal work introducing the concept of adding Gaussian noise iteratively and learning to reverse the process.
    *   **Denoising Diffusion Implicit Models (DDIM) [22]:** Improved sampling speed and determinism for `diffusion models`.
    *   **Score-based Generative Modeling through Stochastic Differential Equations (SDEs) [23]:** Unified `diffusion models` under an `SDE/ODE` framework, providing a way to introduce stochasticity or determinism during sampling. This work is directly relevant to `Flow-GRPO`'s `ODE-to-SDE` conversion.
    *   **Unified Diffusion and Flow Models [28, 29]:** Recent theoretical work that unifies `diffusion` and `flow models` under a common `SDE/ODE` framework, supporting `Flow-GRPO`'s theoretical foundations.
*   **RL for Generative Models:**
    *   **Training Diffusion Models with Reinforcement Learning [12] (`DDPO`):** Applied `RL` to `diffusion models`. `Flow-GRPO` extends this idea to the more efficient `flow matching models` and faces the challenge of their deterministic nature.
    *   **RL for LLMs [10, 11]:** Demonstrated the power of `online RL` (`PPO`, `GRPO`) in enhancing `LLM` reasoning. `Flow-GRPO` seeks to transfer this success to `T2I` models.
    *   **Direct Preference Optimization (DPO) and variants [13, 14, 15, 38, 39]:** `Offline RL` techniques that align models with human preferences. `Flow-GRPO` focuses on `online RL`, which allows for continuous interaction and exploration, and shows it outperforms `online DPO` in some settings.
*   **Alignment for `T2I` Models:** A broad category of methods aimed at improving consistency with human preferences or specific criteria:
    *   **Differentiable Rewards [30, 31, 32, 33]:** Fine-tuning with rewards where gradients can be backpropagated directly. `Flow-GRPO` doesn't require differentiable rewards, allowing for broader applicability.
    *   **Reward Weighted Regression (RWR) [34, 35, 36, 37]:** Techniques that weigh samples by their rewards during fine-tuning.
    *   **PPO-style Policy Gradients [47, 48, 49, 50, 51, 52]:** Other applications of `policy gradient RL` to `T2I` or `diffusion models`.
    *   **Training-free Alignment Methods [53, 54, 55]:** Methods that adjust generation without explicit training.

## 3.3. Technological Evolution

The field of generative imaging has rapidly evolved:
1.  **Early Generative Models (GANs, VAEs):** Capable of generating diverse images but often struggled with fidelity or mode collapse.
2.  **Diffusion Models (`DDPM`, `DDIM`):** Introduced a new paradigm of iterative denoising from noise to data, achieving unprecedented image quality and diversity. Their foundation in `SDEs` provided flexibility in sampling.
3.  **Flow Matching Models (`Rectified Flow`, `Flow Matching`):** Emerged as a more efficient alternative to `diffusion models`, directly learning `velocity fields` and enabling faster, deterministic `ODE-based` sampling while maintaining competitive quality. These models became the backbone of state-of-the-art `T2I` systems like `SD3.5-M` and `FLUX`.
4.  **Alignment with Human Preferences and Instructions:** As generative models improved, the focus shifted to aligning their outputs more precisely with user intentions, `human preferences`, and complex instructions. This led to the adoption of `RL` techniques, initially for `LLMs` and then increasingly for `T2I` models.

    `Flow-GRPO` fits into this timeline by pushing the boundaries of alignment for the most advanced image generative models (flow matching models) by integrating the powerful `online RL` paradigm, which was previously challenging due to the deterministic nature of these models.

## 3.4. Differentiation Analysis

Compared to the main methods in related work, `Flow-GRPO` introduces several core innovations:

*   **Online RL for Flow Matching (First of its Kind):** Previous works applied `RL` primarily to `diffusion models` (e.g., `DDPO` [12]) or used `offline RL` (e.g., `DPO` [14, 39]) for `flow-based models`. `Flow-GRPO` is the *first* to successfully integrate *online policy gradient RL* into the inherently deterministic `flow matching` framework. This is a significant distinction, as `online RL` offers continuous exploration and adaptation that `offline RL` lacks.
*   **ODE-to-SDE Conversion as Key for Stochasticity:** Unlike prior work that might reformulate `velocity prediction` to estimate Gaussian distributions (e.g., [56] for text-to-speech `flow models`, requiring retraining the pre-trained model) or focus on `SDE-based stochasticity` only at inference time [57], `Flow-GRPO` proposes a direct `ODE-to-SDE` conversion that preserves marginal distributions. This allows injecting stochasticity for `RL exploration` into a pre-trained deterministic flow model *without retraining its core components*, making it a plug-and-play solution.
*   **Denoising Reduction for Training Efficiency:** The `Denoising Reduction` strategy is novel in this context. While efficient sampling is generally a goal, this specific technique of using *fewer steps for training data collection but full steps for inference* is crucial for making `online RL` practical for computationally intensive generative models. This allows `Flow-GRPO` to gather low-quality but informative trajectories efficiently, a key enabling factor for `online RL`.
*   **Robust Reward Hacking Prevention via `KL` Regularization:** The paper rigorously demonstrates the effectiveness of `KL` regularization in preventing `reward hacking` (quality degradation, diversity collapse), which is a common challenge in `RL` applications. This is explicitly shown to be superior to simply early stopping and is a critical component for stable, high-quality `RL` fine-tuning.
*   **Generalizability Across Reward Types:** `Flow-GRPO` is shown to be effective across various reward types: `verifiable rule-based rewards` (GenEval, Visual Text Rendering) and `model-based human preference rewards` (PickScore). This suggests a broad applicability of the framework.

    In essence, `Flow-GRPO` innovatively bridges the gap between the efficiency and quality of `flow matching models` and the reasoning/alignment power of `online RL`, overcoming the inherent incompatibilities through clever technical strategies.

# 4. Methodology

## 4.1. Principles

The core idea of `Flow-GRPO` is to enhance `flow matching models` for text-to-image (T2I) generation by leveraging the power of `online reinforcement learning (RL)`. This integration is driven by the principle that `RL` can optimize models for complex, human-defined objectives (like compositional accuracy or human preferences) that are difficult to capture with traditional supervised learning loss functions.

The theoretical basis and intuition behind `Flow-GRPO` can be broken down into two main principles, addressing the key challenges of applying `online RL` to `flow models`:

1.  **Introducing Stochasticity for RL Exploration:** `Online RL` fundamentally relies on `stochastic sampling` to explore the `environment` and learn optimal `policies`. However, standard `flow matching models` are inherently deterministic, generating images by solving `Ordinary Differential Equations (ODEs)`. The principle here is to convert this deterministic `ODE-based` generative process into an equivalent `Stochastic Differential Equation (SDE)` process that preserves the original model's marginal probability distribution at all timesteps. This `ODE-to-SDE` conversion injects the necessary randomness (`exploration noise`) into the generation process, allowing the `RL agent` (the flow model) to try different actions (denoising steps leading to different images) and learn from their `rewards`. The underlying intuition is that while the path from noise to data becomes stochastic, the overall distribution of generated images remains consistent with the pre-trained flow model, ensuring quality while enabling exploration.

2.  **Improving Sampling Efficiency for Online RL Training:** `Online RL` requires collecting many `trajectories` (sequences of states, actions, and rewards) to update the `policy`. `Flow models` typically require numerous iterative `denoising steps` to generate a single high-quality image, making data collection prohibitively slow and expensive for `online RL`. The principle of `Denoising Reduction` is that for the purpose of collecting training data for `RL`, high-fidelity images are not strictly necessary. Instead, "low-quality but still informative trajectories" generated with significantly fewer `denoising steps` can be sufficient to provide a useful `reward signal`. The intuition is that `RL` optimizes based on *relative preferences* (which sample is better than another), and this relative signal can still be extracted even from less refined samples. By drastically cutting the number of steps during training, the wall-clock time for data collection is reduced, making `online RL` practical. The full, high-step schedule is then reserved for inference to ensure top-quality final outputs.

    By adhering to these principles, `Flow-GRPO` aims to bridge the gap between efficient, high-quality image generation and the powerful optimization capabilities of `online RL`.

## 4.2. Core Methodology In-depth (Layer by Layer)

`Flow-GRPO` adapts the `GRPO` algorithm for `flow matching models` by introducing two key strategies: `ODE-to-SDE` conversion for stochasticity and `Denoising Reduction` for efficiency.

### 4.2.1. GRPO on Flow Matching

The overall goal of `RL` is to learn a `policy` $\pi_{\theta}$ (parameterized by $\theta$, which represents the parameters of the flow model's `velocity field` predictor) that maximizes the expected cumulative reward. The paper formulates this with a regularized objective:
\$
\operatorname*{max}_{\theta} \mathbb{E}_{(s_0, a_0, \ldots, s_T, a_T) \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \left( R(s_t, a_t) - \beta D_{\mathrm{KL}}(\pi_{\theta}(\cdot \mid s_t) || \pi_{\mathrm{ref}}(\cdot \mid s_t) ) \right) \right]
\$
Here:
*   $\theta$: Parameters of the `policy` (the flow model).
*   $(s_0, a_0, \ldots, s_T, a_T)$: A `trajectory` of `states` and `actions` sampled according to the `policy` $\pi_{\theta}$.
*   $R(s_t, a_t)$: The `reward` received at `timestep` $t$. In this MDP, rewards are typically given only at the final step (when the image $\pmb{x}_0$ is generated).
*   $\beta$: A hyperparameter controlling the strength of the `KL divergence` regularization.
*   $D_{\mathrm{KL}}(\pi_{\theta}(\cdot \mid s_t) || \pi_{\mathrm{ref}}(\cdot \mid s_t))$: `KL divergence` between the current `policy` $\pi_{\theta}$ and a reference `policy` $\pi_{\mathrm{ref}}$ (typically the `old policy` or the initial pre-trained model) at `state` $s_t$. This regularization term prevents the `policy` from drifting too far from the reference, mitigating `reward hacking` and maintaining stability.

**Denoising as an `MDP`:**
As described in Section 3 of the paper, the iterative `denoising process` in `flow matching models` is framed as an `MDP` $( S , { \mathcal { A } } , \rho _ { 0 } , P , R )$.
*   **State $s_t$:** At `timestep` $t$, the `state` is defined as $\pmb{s}_t \triangleq ( \pmb{c}, t, \pmb{x}_t )$, where $\pmb{c}$ is the text `condition` (prompt), $t$ is the current `timestep`, and $\pmb{x}_t$ is the current noisy image representation.
*   **Action $a_t$:** The `action` is the `denoised sample` $\mathbf{\Phi}_{\pmb{a}_t} \triangleq \pmb{x}_{t-1}$ predicted by the model, representing the image at the previous `timestep` (closer to the clean image).
*   **Policy $\pi(\mathbf{a}_t \mid \pmb{s}_t)$:** The `policy` is $p_{\boldsymbol \theta}(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \boldsymbol{\mathsf{c}})$, which describes the probability distribution over possible next image states $\mathbf{x}_{t-1}$ given the current noisy image $\mathbf{x}_t$ and the text condition $\boldsymbol{\mathsf{c}}$.
*   **Transition $P(\pmb{s}_{t+1} \mid \pmb{s}_t, \pmb{a}_t)$:** This is deterministic, meaning applying action $\pmb{a}_t$ to state $\pmb{s}_t$ always leads to a specific next state $( \delta _ { \pmb{c} } , \delta _ { t-1 } , \delta _ { \pmb{x}_{t-1} } )$. The prompt $\pmb{c}$ remains constant, the `timestep` decreases by 1, and the image becomes $\pmb{x}_{t-1}$.
*   **Initial State Distribution $\rho_0(\pmb{\mathscr{s}}_0)$:** This is $(p(\pmb{c}), \delta_T, \mathcal{N}(\pmb{0}, \mathbf{I}))$, meaning the process starts with a randomly sampled prompt $\pmb{c}$, at the maximum `timestep` $T$, and with an initial noisy image $\pmb{x}_T$ sampled from a standard Gaussian distribution $\mathcal{N}(\pmb{0}, \mathbf{I})$.
*   **Reward $R(\pmb{s}_t, \pmb{a}_t)$:** The `reward` is sparse, given only at the final step when $t=0$, i.e., $R(\pmb{s}_t, \pmb{a}_t) \triangleq r(\pmb{x}_0, \pmb{c})$ if $t=0$, and `0` otherwise. This $r(\pmb{x}_0, \pmb{c})$ is the task-specific reward (e.g., GenEval score, OCR accuracy, PickScore).

**`GRPO` Advantage Estimation:**
`GRPO` [16] uses a group relative formulation for estimating the advantage. Given a prompt $\pmb{c}$, the flow model samples a group of $G$ individual images $\{ \boldsymbol{x}_0^i \}_{i=1}^G$ and their corresponding trajectories $\{ ( \pmb{x}_T^i, \pmb{x}_{T-1}^i, \ldots, \pmb{x}_0^i ) \}_{i=1}^G$. The advantage $\hat{A}_t^i$ for the $i$-th image at `timestep` $t$ is calculated by normalizing the group-level rewards:
\$
\hat{A}_t^i = \frac{R(\pmb{x}_0^i, \pmb{c}) - \mathrm{mean}(\{R(\pmb{x}_0^i, \pmb{c})\}_{i=1}^G)}{\mathrm{std}(\{R(\pmb{x}_0^i, \pmb{c})\}_{i=1}^G)}
\$
Here:
*   $R(\pmb{x}_0^i, \pmb{c})$: The final `reward` for the $i$-th generated image $\pmb{x}_0^i$ given prompt $\pmb{c}$.
*   $\mathrm{mean}(\cdot)$ and $\mathrm{std}(\cdot)$: The mean and standard deviation of the `rewards` across all $G$ images in the group for the same prompt.
    This normalization makes the advantage estimate robust to the absolute scale of rewards and focuses on relative performance within a group.

**`GRPO` Objective:**
`GRPO` optimizes the policy model by maximizing the following objective:
\$
\mathcal{L}_{\mathrm{Flow-GRPO}}(\theta) = \mathbb{E}_{c \sim \mathcal{C}, \{ \boldsymbol{x}^i \}_{i=1}^G \sim \pi_{\theta_{\mathrm{old}}}(\cdot \vert c)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left( \operatorname{min}\left( r_t^i(\theta) \hat{A}_t^i, \mathrm{clip}\Big( r_t^i(\theta), 1-\varepsilon, 1+\varepsilon \Big) \hat{A}_t^i \right) - \beta D_{\mathrm{KL}}(\pi_{\theta} || \pi_{\mathrm{ref}}) \right) \right]
\$
where the `probability ratio` $r_t^i(\theta)$ is:
\$
r_t^i(\theta) = \frac{p_{\theta}(x_{t-1}^i \mid x_t^i, c)}{p_{\theta_{\mathrm{old}}}(x_{t-1}^i \mid x_t^i, c)}
\$
And:
*   $\mathcal{C}$: Distribution of prompts.
*   $\theta_{\mathrm{old}}$: Parameters of the `policy` used to collect the current batch of samples (the `old policy`), which is periodically updated to $\theta$.
*   $\varepsilon$: A small clipping parameter (similar to `PPO` [20]) that limits the magnitude of `policy` updates, ensuring stability.
*   $\beta$: The `KL regularization` coefficient, as explained earlier.
    This objective aims to increase the probability of actions that lead to higher-than-average rewards (positive advantage) and decrease the probability of actions leading to lower-than-average rewards (negative advantage), while keeping the `policy` close to the `old policy` and preventing excessive divergence.

### 4.2.2. From ODE to SDE

The deterministic nature of `flow matching models` (based on `ODEs`) presents two problems for `GRPO`:
1.  Computing the `probability ratio` $r_t^i(\theta) = \frac{p_{\theta}(x_{t-1}^i \mid x_t^i, c)}{p_{\theta_{\mathrm{old}}}(x_{t-1}^i \mid x_t^i, c)}$ is computationally expensive under deterministic dynamics due to divergence estimation.
2.  More critically, `RL` relies on `exploration` through `stochastic sampling`. Deterministic sampling lacks the randomness needed for `RL` to explore different outcomes and learn.

    To address this, the paper converts the deterministic Flow-ODE into an equivalent `SDE` that matches the original model's marginal probability density function at all timesteps.

**Original ODE:**
The standard `flow matching ODE` is given by:
\$
\mathrm{d}\pmb{x}_t = \pmb{v}_t \mathrm{d}t
\$
where $\pmb{v}_t$ is the `velocity field` learned via the `flow matching objective`. This `ODE` implies a one-to-one mapping between successive `timesteps`.

**Generic SDE and Fokker-Planck Equation:**
A generic `SDE` has the form:
\$
\mathrm{d}\pmb{x}_t = f_{\mathrm{SDE}}(\pmb{x}_t, t)\mathrm{d}t + \sigma_t \mathrm{d}\pmb{w}
\$
where:
*   $f_{\mathrm{SDE}}(\pmb{x}_t, t)$: The `drift coefficient`.
*   $\sigma_t$: The `diffusion coefficient` controlling the level of stochasticity.
*   $\mathrm{d}\pmb{w}$: Increments of a `Wiener process` (standard Brownian motion).

    The `marginal probability density` $p_t(\pmb{x})$ of an `SDE` evolves according to the `Fokker-Planck equation` [74]:
\$
\partial_t p_t(x) = - \nabla \cdot [ f_{\mathrm{SDE}}(\pmb{x}_t, t) p_t(\pmb{x}) ] + \frac{1}{2} \nabla^2 [ \sigma_t^2 p_t(\pmb{x}) ]
\$
For the deterministic `ODE` (Eq. 10), its `marginal probability density` evolves as:
\$
\partial_t p_t(\pmb{x}) = - \nabla \cdot [ \pmb{v}_t(\pmb{x}_t, t) p_t(\pmb{x}) ]
\$

**Equating Marginal Distributions:**
To ensure the `SDE` shares the same `marginal distribution` as the `ODE`, their `Fokker-Planck equations` must be equal:
\$
- \nabla \cdot [ f_{\mathrm{SDE}} p_t(\pmb{x}) ] + \frac{1}{2} \nabla^2 [ \sigma_t^2 p_t(\pmb{x}) ] = - \nabla \cdot [ \pmb{v}_t(\pmb{x}_t, t) p_t(\pmb{x}) ]
  \$
Using the identity $\nabla^2 [ \sigma_t^2 p_t(\pmb{x}) ] = \sigma_t^2 \nabla \cdot ( p_t(\pmb{x}) \nabla \log p_t(\pmb{x}) )$, and after substituting and simplifying (detailed in Appendix A), the `drift coefficient` $f_{\mathrm{SDE}}$ is derived as:
\$
f_{\mathrm{SDE}} = \boldsymbol{v}_t(\boldsymbol{x}_t, t) + \frac{\sigma_t^2}{2} \nabla \log p_t(\boldsymbol{x})
\$
This leads to the `forward SDE` with the desired `marginal distribution`:
\$
\mathrm{d}\pmb{x}_t = \bigg( \pmb{v}_t(\pmb{x}_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(\pmb{x}_t) \bigg)\mathrm{d}t + \sigma_t \mathrm{d}\pmb{w}
\$
Here, $\nabla \log p_t(\pmb{x}_t)$ is the `score function`.

**Reverse-Time SDE for Sampling:**
For practical sampling, a `reverse-time SDE` is needed, which runs from the final state back to the initial state. The relationship between `forward` and `reverse-time SDEs` is established by [75, 23]. If a `forward SDE` is $\mathrm{d}\pmb{x}_t = f(\pmb{x}_t, t)\mathrm{d}t + g(t)\mathrm{d}\pmb{w}$, its `reverse-time SDE` is:
\$
\mathrm{d}\pmb{x}_t = \left[ f(\pmb{x}_t, t) - g^2(t) \nabla \log p_t(\pmb{x}_t) \right]\mathrm{d}t + g(t)\mathrm{d}\overline{\pmb{w}}
\$
Setting $g(t) = \sigma_t$ and substituting $f(\pmb{x}_t, t)$ from Eq. 17, we get the `reverse-time SDE`:
\$
\mathrm{d}\pmb{x}_t = \bigg[ \pmb{v}_t(\pmb{x}_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(\pmb{x}_t) - \sigma_t^2 \nabla \log p_t(\pmb{x}_t) \bigg]\mathrm{d}t + \sigma_t \mathrm{d}\overline{\pmb{w}}
\$
This simplifies to:
\$
\mathrm{d}\pmb{x}_t = \left( \pmb{v}_t(\pmb{x}_t) - \frac{\sigma_t^2}{2} \nabla \log p_t(\pmb{x}_t) \right) \mathbf{d}t + \sigma_t \mathbf{d}\pmb{w}
\$
The term $\nabla \log p_t(\pmb{x}_t)$ is implicitly linked to the `velocity field` $\pmb{v}_t$. For the `Rectified Flow` framework used in the paper, the authors use the linear interpolation $\pmb{x}_t = (1-t)\pmb{x}_0 + t\pmb{x}_1$, where $\alpha_t = 1-t$ and $\beta_t = t$.
From this, the `conditional score` is $\nabla \log p_{t|0}(\pmb{x}_t | \pmb{x}_0) = - \frac{\pmb{x}_1}{\beta_t}$.
The `marginal score` becomes $\nabla \log p_t(\pmb{x}_t) = - \frac{1}{\beta_t} \mathbb{E}[\pmb{x}_1 \mid \pmb{x}_t]$.
After a series of derivations (Equations 22-26 in Appendix A), the `score function` is expressed in terms of $\pmb{x}_t$ and $\pmb{v}_t(\pmb{x}_t)$:
\$
\nabla \log p_t(\pmb{x}) = - \frac{\pmb{x}}{t} - \frac{1-t}{t} \pmb{v}_t(\pmb{x})
\$
Substituting this `score function` back into the `reverse-time SDE` (Eq. 21) yields the final `SDE` for `Rectified Flow`:
\$
\mathrm{d}\pmb{x}_t = \left[ \pmb{v}_t(\pmb{x}_t) + \frac{\sigma_t^2}{2t} \left( \pmb{x}_t + (1-t) \pmb{v}_t(\pmb{x}_t) \right) \right] \mathrm{d}t + \sigma_t \mathrm{d}\pmb{w}
\$
This is the `SDE` that the `Flow-GRPO` model will sample from. To numerically solve this `SDE`, `Euler-Maruyama discretization` is applied, resulting in the following update rule:
\$
\boxed{x_{t+\Delta t} = x_t + \left[ v_{\theta}(x_t, t) + \frac{\sigma_t^2}{2t} \big( x_t + (1-t) v_{\theta}(x_t, t) \big) \right] \Delta t + \sigma_t \sqrt{\Delta t} \epsilon}
\$
Here:
*   $x_t$: The image representation at `timestep` $t$.
*   $v_{\theta}(x_t, t)$: The `velocity field` predicted by the model (parameterized by $\theta$) at `state` $x_t$ and `timestep` $t$.
*   $\sigma_t$: The `diffusion coefficient`, which controls the level of stochasticity. The paper uses $\sigma_t = a \sqrt{\frac{t}{1-t}}$, where $a$ is a scalar hyper-parameter.
*   $\Delta t$: The `timestep` size for discretization.
*   $\epsilon \sim \mathcal{N}(0, I)$: A sample from a standard Gaussian distribution, explicitly injecting `stochasticity` into the sampling process.

    This `SDE` update rule defines the `policy` $\pi_{\theta}(x_{t+\Delta t} \mid x_t, c)$, which is an `isotropic Gaussian distribution`. This allows for a closed-form computation of the `KL divergence` between $\pi_{\theta}$ and the `reference policy` $\pi_{\mathrm{ref}}$ (which would be based on $v_{\mathrm{ref}}$):
\$
D_{\mathrm{KL}}(\pi_{\theta} || \pi_{\mathrm{ref}}) = \frac{||\overline{x}_{t+\Delta t, \theta} - \overline{x}_{t+\Delta t, \mathrm{ref}}||^2}{2\sigma_t^2 \Delta t} = \frac{\Delta t}{2} \left( \frac{\sigma_t (1-t)}{2t} + \frac{1}{\sigma_t} \right)^2 ||v_{\theta}(x_t, t) - v_{\mathrm{ref}}(x_t, t)||^2
\$
Here:
*   $\overline{x}_{t+\Delta t, \theta}$: The mean of the distribution for $x_{t+\Delta t}$ under $\pi_{\theta}$.
*   $\overline{x}_{t+\Delta t, \mathrm{ref}}$: The mean of the distribution for $x_{t+\Delta t}$ under $\pi_{\mathrm{ref}}$.
*   This formula highlights that the `KL divergence` is proportional to the squared difference between the `velocity fields` of the current and `reference policies`, scaled by terms related to $\sigma_t$ and $\Delta t$. This makes the `KL` regularization directly influence the similarity of the learned `velocity field` to the reference.

### 4.2.3. Denoising Reduction

To address the high computational cost of data collection for `online RL`, the `Denoising Reduction` strategy is employed:
*   **Training Phase:** During `online RL` training, the model uses significantly fewer `denoising steps` (e.g., $T=10$) to generate samples. These samples, while of lower visual quality, are sufficient to provide a useful `reward signal` for `GRPO`'s relative advantage estimation. This drastically reduces the time and resources needed for data collection.
*   **Inference Phase:** For generating final, high-quality images during evaluation or deployment, the model reverts to its original, full `denoising steps` (e.g., $T=40$ for `SD3.5-M`).

    This strategy allows for faster `RL` training without compromising the quality of the final outputs, as the underlying flow model is still capable of high-fidelity generation when given enough steps.

# 5. Experimental Setup

## 5.1. Datasets

The experiments evaluate `Flow-GRPO` across three main tasks, each with specific prompt generation and reward definitions:

### 5.1.1. Compositional Image Generation
*   **Dataset Source:** The GenEval [17] benchmark.
*   **Characteristics:** This benchmark assesses `T2I` models on complex compositional prompts that require precise understanding and generation of:
    *   **Object Counting:** e.g., "three red apples."
    *   **Spatial Relations:** e.g., "a cat on the roof of a house."
    *   **Attribute Binding:** e.g., "a blue car and a red car."
*   **Prompt Generation:** Training prompts are generated using official GenEval scripts, which employ templates and random combinations to create a diverse prompt dataset. The test set is strictly deduplicated to avoid overlap with training data, treating prompts differing only in object order as identical.
*   **Prompt Ratio:** Based on the base model's initial accuracy, the ratio of prompt types used for training is: Position : Counting : Attribute Binding : Colors : Two Objects : Single Object = `7 : 5 : 3 : 1 : 1 : 0`. This prioritizes more challenging compositional aspects.
*   **Example Data Sample (GenEval-style prompt):** "a photo of a blue pizza and a yellow baseball glove." (As seen in Figure 24 from the appendix).

### 5.1.2. Visual Text Rendering
*   **Dataset Source:** Prompts generated by `GPT-4o`.
*   **Characteristics:** This task evaluates the model's ability to accurately render specified text within an image.
*   **Prompt Generation:** Each prompt follows the template `A sign that says "text"`. The placeholder `"text"` is the exact string the model should render. 20K training prompts and 1K test prompts were generated by `GPT-4o`.
*   **Example Data Sample (Visual Text Rendering prompt):** `A sign that says "caution: telepathic subjects"` (As seen in Figure 25 from the appendix).

### 5.1.3. Human Preference Alignment
*   **Reward Model Source:** PickScore [19].
*   **Characteristics:** This task aims to align `T2I` models with general human aesthetic and semantic preferences.
*   **Prompt Generation:** The paper uses prompts from various sources to train for human preference alignment.
*   **Example Data Sample (PickScore-style prompt):** "a woman on top of a horse" (As seen in Figure 28 from the appendix).

## 5.2. Evaluation Metrics

For every evaluation metric, the following structure is provided:

### 5.2.1. Task-Specific Metrics

*   **GenEval Accuracy (Compositional Image Generation):**
    1.  **Conceptual Definition:** Measures how accurately the generated image reflects complex compositional elements specified in the text prompt, such as correct object counts, colors, and spatial relationships. It's often assessed by detecting objects and analyzing their attributes and arrangements.
    2.  **Mathematical Formula:** The reward function $r$ directly serves as the accuracy metric for GenEval tasks.
        *   **Counting:**
            \$
            r = 1 - \frac{|N_{\mathrm{gen}} - N_{\mathrm{ref}}|}{\bar{N}_{\mathrm{ref}}}
            \$
        *   **Position / Color:**
            If object count is correct, a partial reward is given. The remaining reward is granted if the predicted position or color is also correct.
    3.  **Symbol Explanation:**
        *   $N_{\mathrm{gen}}$: Number of objects generated by the model.
        *   $N_{\mathrm{ref}}$: Number of objects referenced in the prompt.
        *   $\bar{N}_{\mathrm{ref}}$: (Implied from the context, typically) The reference count or an average/expected reference count for normalization.
*   **OCR Accuracy (Visual Text Rendering):**
    1.  **Conceptual Definition:** Quantifies the accuracy of text rendered within the generated image compared to the target text specified in the prompt. It's based on the minimum changes needed to transform the rendered text into the target text.
    2.  **Mathematical Formula:**
        \$
        r = \mathrm{max}\left(1 - \frac{N_{\mathrm{e}}}{N_{\mathrm{ref}}}, 0\right)
        \$
    3.  **Symbol Explanation:**
        *   $N_{\mathrm{e}}$: The minimum `edit distance` (e.g., Levenshtein distance) between the text rendered in the image and the target text from the prompt.
        *   $N_{\mathrm{ref}}$: The number of characters in the target text (the string within quotation marks in the prompt).
*   **PickScore (Human Preference Alignment):**
    1.  **Conceptual Definition:** A `model-based reward` that predicts human preferences for `T2I` generated images. It's trained on a large dataset of `human-annotated pairwise comparisons` of images from the same prompt and provides an overall score reflecting prompt alignment and visual quality.
    2.  **Mathematical Formula:** PickScore is typically a neural network model, so there isn't a simple single formula. It outputs a scalar score, $S = \mathrm{PickScore}(\mathrm{image}, \mathrm{prompt})$.
    3.  **Symbol Explanation:**
        *   $\mathrm{image}$: The generated image.
        *   $\mathrm{prompt}$: The text prompt used for generation.
        *   $S$: A scalar score indicating the model's predicted human preference for the `image-prompt` pair.

### 5.2.2. Image Quality & Preference Metrics (for Reward Hacking Detection)

To detect `reward hacking` (where task-specific reward increases but general image quality or diversity declines), the paper uses several automatic image quality metrics, all computed on `DrawBench` [1], a comprehensive benchmark with diverse prompts.

*   **Aesthetic Score [59]:**
    1.  **Conceptual Definition:** A metric that predicts the perceived aesthetic quality of an image, typically trained on human aesthetic ratings. It aims to capture subjective beauty.
    2.  **Mathematical Formula:** It is a `CLIP-based linear regressor`. The formula is typically not published as a simple equation but represents the output of a trained model:
        \$
        S_{\mathrm{Aesthetic}} = \mathrm{Regressor}(\mathrm{CLIP\_Features}(\mathrm{image}))
        \$
    3.  **Symbol Explanation:**
        *   $\mathrm{image}$: The input image.
        *   $\mathrm{CLIP\_Features}(\mathrm{image})$: Feature embeddings extracted from the image using a pre-trained `CLIP` model.
        *   $\mathrm{Regressor}$: A linear regression model trained to map `CLIP features` to aesthetic scores.
        *   $S_{\mathrm{Aesthetic}}$: The predicted aesthetic score.
*   **DeQA score [60]:**
    1.  **Conceptual Definition:** A `multimodal large language model (MLLM)`-based image quality assessment (`IQA`) model. It quantifies how distortions, texture damage, and other low-level artifacts affect perceived quality, providing a more objective measure of image fidelity.
    2.  **Mathematical Formula:** Similar to `PickScore`, `DeQA` is a complex neural network. Its output is a scalar score, $S = \mathrm{DeQA}(\mathrm{image})$.
    3.  **Symbol Explanation:**
        *   $\mathrm{image}$: The input image.
        *   $S$: A scalar score representing the image's quality in terms of distortions and artifacts.
*   **ImageReward [32]:**
    1.  **Conceptual Definition:** A general-purpose `T2I human preference reward model` that evaluates multiple criteria, including `text-image alignment`, `visual fidelity`, and `harmlessness`.
    2.  **Mathematical Formula:** `ImageReward` is a deep neural network that outputs a scalar score, $S = \mathrm{ImageReward}(\mathrm{image}, \mathrm{prompt})$.
    3.  **Symbol Explanation:**
        *   $\mathrm{image}$: The generated image.
        *   $\mathrm{prompt}$: The text prompt.
        *   $S$: A scalar score reflecting human preference based on alignment, fidelity, and harmlessness.
*   **UnifiedReward [61]:**
    1.  **Conceptual Definition:** A recently proposed unified reward model designed for `multimodal understanding and generation`, aiming to achieve state-of-the-art performance in `human preference assessment`. It is intended to be a comprehensive measure of overall quality and alignment.
    2.  **Mathematical Formula:** `UnifiedReward` is also a complex neural network, producing a scalar score, $S = \mathrm{UnifiedReward}(\mathrm{image}, \mathrm{prompt})$.
    3.  **Symbol Explanation:**
        *   $\mathrm{image}$: The generated image.
        *   $\mathrm{prompt}$: The text prompt.
        *   $S$: A scalar score representing a unified measure of multimodal understanding and generation quality.
*   **Diversity Score:** (Implicitly measured through qualitative assessment and sometimes quantitative metrics like `FID` or `CLIP Score` distribution, though not a standalone formula given here.)
    1.  **Conceptual Definition:** Measures the variety and range of outputs generated by a model for a given prompt or set of prompts. A high diversity score indicates the model can produce distinct and varied images, while low diversity might suggest mode collapse.
    2.  **Mathematical Formula:** Not explicitly provided with a standalone formula in the paper for diversity, but typically assessed via metrics like `FID` (Fréchet Inception Distance), `CLIP Score` distribution width, or qualitative observation of generated samples. In Table 6, `CLIP Score` is used, where a higher score implies better `text-image alignment`, and `Diversity Score` is explicitly reported, likely derived from the spread of embeddings.
    3.  **Symbol Explanation:** Not applicable for a generic formula, but in the context of Table 6, `CLIP Score ↑` indicates that a higher score is better for text-image alignment, and `Diversity Score ↑` indicates higher scores are better for diversity.

## 5.3. Baselines

`Flow-GRPO` was compared against several representative alignment methods, categorized by their approach:

1.  **Supervised Fine-Tuning (SFT):**
    *   **Description:** This baseline selects the highest-reward image within each group of generated images and fine-tunes the model on it using standard supervised learning objectives.
    *   **Representativeness:** Represents a straightforward, direct optimization approach based on explicit high-quality samples.
2.  **Flow-DPO [14, 39] (Direct Preference Optimization):**
    *   **Description:** An `offline RL` technique that uses pairwise preferences. For each group of generated images, the highest-reward image is designated as the "chosen" sample, and the lowest-reward image as the "rejected" sample. The `DPO loss` is then applied to these pairs.
    *   **Representativeness:** A prominent `offline RL` method widely used for alignment tasks, particularly in `LLMs` and increasingly in `generative models`.
3.  **Flow-RWR [14, 76] (Reward Weighted Regression):**
    *   **Description:** An `online reward-weighted regression` method that applies a `softmax` over rewards within each group and performs `reward-weighted likelihood maximization`. It guides the model to prioritize high-reward regions.
    *   **Representativeness:** A class of `RL` methods that use rewards to weight training samples, common for fine-tuning.
4.  **Online Variants (of SFT, Flow-DPO, Flow-RWR):**
    *   **Description:** The "online" versions of the above methods update their data collection models (the policies generating samples for training) every 40 steps, reflecting an adaptive learning process, similar to `Flow-GRPO`.
    *   **Representativeness:** Crucial for a fair comparison against `Flow-GRPO`, which is an `online RL` method itself.
5.  **DDPO [12] (Training Diffusion Models with Reinforcement Learning):**
    *   **Description:** An `online RL` method originally developed for `diffusion-based backbones`. The paper adapted it to `flow-matching models` using the `ODE-to-SDE` conversion for comparison.
    *   **Representativeness:** A direct `RL` competitor for generative models, specifically diffusion, and thus relevant for showing `Flow-GRPO`'s advantages on `flow models`.
6.  **ReFL [32] (Reward-guided Fine-tuning of Latent Diffusion):**
    *   **Description:** Directly fine-tunes `diffusion models` by viewing `reward model scores` as `human preference losses` and `back-propagating gradients` to a randomly picked late `timestep`.
    *   **Representativeness:** Another `RL-like` alignment method that uses differentiable rewards.
7.  **ORW [35] (Online Reward-Weighted Regression):**
    *   **Description:** An `online reward-weighted regression` method that uses `Wasserstein-2 regularization` to prevent `policy collapse` and maintain diversity, differing from `KL regularization`.
    *   **Representativeness:** A distinct `online RL` approach that addresses `policy collapse` using a different regularization technique than `Flow-GRPO`.

        These baselines collectively cover various strategies for aligning `T2I` models, including supervised approaches, `offline RL`, and other `online RL` variants, allowing `Flow-GRPO` to be evaluated comprehensively.

# 6. Results & Analysis

## 6.1. Core Results Analysis

The experimental results strongly validate `Flow-GRPO`'s effectiveness across multiple text-to-image tasks, demonstrating significant improvements in compositional generation, text rendering, and human preference alignment, all while maintaining image quality and diversity.

**Overall Performance and `Reward Hacking` Mitigation:**
Figure 1 (from the original paper) provides a high-level overview:
*   **(a) GenEval performance rises steadily throughout `Flow-GRPO`'s training and outperforms `GPT-4o`.**: This highlights the primary success in compositional tasks.
*   **(b) Image quality metrics on `DrawBench` [1] remain essentially unchanged.** This is crucial, indicating that `Flow-GRPO` achieves its task-specific gains without sacrificing general image quality, effectively mitigating `reward hacking`.
*   **(c) Human Preference Scores on `DrawBench` improves after training.** This shows the method can also align with broader aesthetic and preference objectives.

    The following figure (Figure 1 from the original paper) summarizes `Flow-GRPO`'s overall performance:

    ![Figure 1: (a) GenEval performance rises steadily throughout Flow-GRPO's training and outperforms GPT-4o. (b) Image quality metrics on DrawBench \[1\] remain essentially unchanged. (c) Human Preference Scores on DrawBench improves after training. Results show that Flow-GRPO enhances the desired capability while preserving image quality and exhibiting minimal reward-hacking.](images/1.jpg)

    **Compositional Image Generation (GenEval):**
`Flow-GRPO` significantly boosts `SD3.5-M`'s ability to handle complex compositional prompts.
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Overall</th>
<th rowspan="2">Single Obj.</th>
<th rowspan="2">Two Obj.</th>
<th rowspan="2">Counting</th>
<th rowspan="2">Colors</th>
<th rowspan="2">Position</th>
<th rowspan="2">Attr. Binding</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8">Diffusion Models</td>
</tr>
<tr>
<td>LDM [62]</td>
<td>0.37</td>
<td>0.92</td>
<td>0.29</td>
<td>0.23</td>
<td>0.70</td>
<td>0.02</td>
<td>0.05</td>
</tr>
<tr>
<td>SD1.5 [62]</td>
<td>0.43</td>
<td>0.97</td>
<td>0.38</td>
<td>0.35</td>
<td>0.76</td>
<td>0.04</td>
<td>0.06</td>
</tr>
<tr>
<td>SD2. [62]</td>
<td>0.50</td>
<td>0.98</td>
<td>0.51</td>
<td>0.44</td>
<td>0.85</td>
<td>0.07</td>
<td>0.17</td>
</tr>
<tr>
<td>SD-XL [63]</td>
<td>0.55</td>
<td>0.98</td>
<td>0.74</td>
<td>0.39</td>
<td>0.85</td>
<td>0.15</td>
<td>0.23</td>
</tr>
<tr>
<td>DALLE-2 [64]</td>
<td>0.52</td>
<td>0.94</td>
<td>0.66</td>
<td>0.49</td>
<td>0.77</td>
<td>0.10</td>
<td>0.19</td>
</tr>
<tr>
<td>DALLE-3 [65]</td>
<td>0.67</td>
<td>0.96</td>
<td>0.87</td>
<td>0.47</td>
<td>0.83</td>
<td>0.43</td>
<td>0.45</td>
</tr>
<tr>
<td colspan="8">Autoregressive Models</td>
</tr>
<tr>
<td>Show-o [66]</td>
<td>0.53</td>
<td>0.95</td>
<td>0.52</td>
<td>0.49</td>
<td>0.82</td>
<td>0.11</td>
<td>0.28</td>
</tr>
<tr>
<td>Emu3-Gen [67]</td>
<td>0.54</td>
<td>0.98</td>
<td>0.71</td>
<td>0.34</td>
<td>0.81</td>
<td>0.17</td>
<td>0.21</td>
</tr>
<tr>
<td>JanusFlow [68]</td>
<td>0.63</td>
<td>0.97</td>
<td>0.59</td>
<td>0.45</td>
<td>0.83</td>
<td>0.53</td>
<td>0.42</td>
</tr>
<tr>
<td>Janus-Pro-7B [69]</td>
<td>0.80</td>
<td>0.99</td>
<td>0.89</td>
<td>0.59</td>
<td>0.90</td>
<td>0.79</td>
<td>0.66</td>
</tr>
<tr>
<td>GPT-4o [18]</td>
<td>0.84</td>
<td>0.99</td>
<td>0.92</td>
<td>0.85</td>
<td>0.92</td>
<td>0.75</td>
<td>0.61</td>
</tr>
<tr>
<td colspan="8">Flow Matching Models</td>
</tr>
<tr>
<td>FLUX.1 Dev [5]</td>
<td>0.66</td>
<td>0.98</td>
<td>0.81</td>
<td>0.74</td>
<td>0.79</td>
<td>0.22</td>
<td>0.45</td>
</tr>
<tr>
<td>SD3.5-L [4]</td>
<td>0.71</td>
<td>0.98</td>
<td>0.89</td>
<td>0.73</td>
<td>0.83</td>
<td>0.34</td>
<td>0.47</td>
</tr>
<tr>
<td>SANA-1.5 4.8B [70]</td>
<td>0.81</td>
<td>0.99</td>
<td>0.93</td>
<td>0.86</td>
<td>0.84</td>
<td>0.59</td>
<td>0.65</td>
</tr>
<tr>
<td>SD3.5-M [4]</td>
<td>0.63</td>
<td>0.98</td>
<td>0.78</td>
<td>0.50</td>
<td>0.81</td>
<td>0.24</td>
<td>0.52</td>
</tr>
<tr>
<td>SD3.5-M+Flow-GRPO</td>
<td style="color: blue;">0.95</td>
<td style="color: blue;">1.00</td>
<td style="color: blue;">0.99</td>
<td style="color: blue;">0.95</td>
<td style="color: blue;">0.92</td>
<td style="color: blue;">0.99</td>
<td style="color: blue;">0.86</td>
</tr>
</tbody>
</table>

As shown in Table 1, `SD3.5-M` with `Flow-GRPO` achieved an outstanding `Overall` GenEval score of `0.95`, a substantial increase from the base `SD3.5-M`'s `0.63`. This score is not only the best among all models listed (including `Diffusion Models`, `Autoregressive Models`, and other `Flow Matching Models`), but it also significantly outperforms `GPT-4o` (`0.84`), which was previously a strong performer.
The improvements are consistent across all sub-tasks, particularly in `Counting` ($0.50 \to 0.95$), `Position` ($0.24 \to 0.99$), and `Attribute Binding` ($0.52 \to 0.86$), which are known challenges for `T2I` models. This indicates `Flow-GRPO`'s ability to learn fine-grained control and reasoning.
Figure 3 from the original paper provides qualitative comparisons on the GenEval benchmark, further illustrating `Flow-GRPO`'s superior performance in `Counting`, `Colors`, `Attribute Binding`, and `Position`. For example, `Flow-GRPO` correctly generates the specified number of objects and their attributes, where the base `SD3.5-M` often fails.
The following figure (Figure 3 from the original paper) visually compares `Flow-GRPO`'s qualitative performance on the GenEval benchmark:

![Figure 3: Qualitative Comparison on the GenEval Benchmark. Our approach demonstrates superior performance in Counting, Colors, Attribute Binding, and Position.](images/3.jpg)

**Visual Text Rendering and Human Preference Alignment:**
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">Task Metric</th>
<th colspan="2">Image Quality</th>
<th colspan="3">Preference Score</th>
</tr>
<tr>
<th>GenEval</th>
<th>OCR Acc.</th>
<th>PickScore</th>
<th>Aesthetic</th>
<th>DeQA</th>
<th>ImgRwd</th>
<th>PickScore</th>
<th>UniRwd</th>
</tr>
</thead>
<tbody>
<tr>
<td>SD3.5-M</td>
<td>0.63</td>
<td>0.59</td>
<td>21.72</td>
<td>5.39</td>
<td>4.07</td>
<td>0.87</td>
<td>22.34</td>
<td>3.33</td>
</tr>
<tr>
<td colspan="9">Compositional Image Generation</td>
</tr>
<tr>
<td>Flow-GRPO (w/o KL)</td>
<td>0.95</td>
<td></td>
<td></td>
<td>4.93</td>
<td>2.77</td>
<td>0.44</td>
<td>21.16</td>
<td>2.94</td>
</tr>
<tr>
<td>Flow-GRPO (w/KL)</td>
<td>0.95</td>
<td></td>
<td></td>
<td>5.25</td>
<td>4.01</td>
<td>1.03</td>
<td>22.37</td>
<td>3.51</td>
</tr>
<tr>
<td colspan="9">Visual Text Rendering</td>
</tr>
<tr>
<td>Flow-GRPO (w/o KL)</td>
<td></td>
<td>0.93</td>
<td></td>
<td>5.13</td>
<td>3.66</td>
<td>0.58</td>
<td>21.79</td>
<td>3.15</td>
</tr>
<tr>
<td>Flow-GRPO (w/KL)</td>
<td></td>
<td>0.92</td>
<td></td>
<td>5.32</td>
<td>4.06</td>
<td>0.95</td>
<td>22.44</td>
<td>3.42</td>
</tr>
<tr>
<td colspan="9">Human Preference Alignment</td>
</tr>
<tr>
<td>Flow-GRPO (w/o KL)</td>
<td></td>
<td></td>
<td>23.41</td>
<td>6.15</td>
<td>4.16</td>
<td>1.24</td>
<td>23.56</td>
<td>3.57</td>
</tr>
<tr>
<td>Flow-GRPO (w/ KL)</td>
<td></td>
<td></td>
<td>23.31</td>
<td>5.92</td>
<td>4.22</td>
<td>1.28</td>
<td>23.53</td>
<td>3.66</td>
</tr>
</tbody>
</table>

Table 2 confirms these gains and further highlights the role of `KL regularization`:
*   **Visual Text Rendering:** `Flow-GRPO (w/KL)` increases `OCR Acc.` from `0.59` to `0.92`. Crucially, `Aesthetic`, `DeQA`, `ImageReward`, `PickScore`, and `UnifiedReward` metrics remain stable or slightly improve, demonstrating that `Flow-GRPO` enhances text rendering without compromising general image quality.
*   **Human Preference Alignment:** `Flow-GRPO (w/KL)` improves `PickScore` (task metric) from `21.72` to `23.31`. Again, general quality metrics are preserved.
*   **Impact of `KL Regularization`:** Comparing `Flow-GRPO (w/o KL)` with `Flow-GRPO (w/KL)` clearly shows the importance of `KL`. Without `KL`, `Image Quality` (e.g., `DeQA` drops from `4.07` to `2.77` for compositional generation) and `Preference Scores` (e.g., `ImageReward` drops from `0.87` to `0.44`) significantly degrade, even if task metrics are high. This is a clear indication of `reward hacking`. The `KL` constraint effectively mitigates this.

**Comparison with Other Alignment Methods:**
Figure 4 from the original paper compares `Flow-GRPO` with various `online` and `offline` alignment methods on the `Compositional Generation Task`.
The following figure (Figure 4 from the original paper) shows the comparison with other alignment methods:

![Figure 4: Comparison with Other Alignment Methods on the Compositional Generation Task.](images/4.jpg)
*该图像是图表，展示了不同对齐方法在组合生成任务中的 GenEval 评分对比。随着训练提示数量的增加，Flow-GRPO 方法的 GenEval 评分显著提高，最高达到 `0.9` 以上，而其他方法的表现有所不同。*

`Flow-GRPO` consistently outperforms all baselines (SFT, Flow-DPO, Flow-RWR, and their online variants) by a significant margin in terms of GenEval score. For instance, `Flow-GRPO` reaches over `0.9`, while the next best `Online DPO` struggles to pass `0.8`. This indicates the superior effectiveness of `online policy gradient` with `GRPO` for `flow matching models`.

## 6.2. Ablation Studies / Parameter Analysis

The paper conducts several ablation studies to understand the behavior and robustness of `Flow-GRPO`'s key components.

### 6.2.1. Reward Hacking and `KL Regularization`
The impact of `KL regularization` is a critical finding:
*   **Observation:** Without the `KL` constraint (`Flow-GRPO (w/o KL)`), models achieve high task-specific rewards but suffer from `quality degradation` (for GenEval and OCR) and `diversity decline` (for PickScore). For example, in Table 2, `DeQA` scores drop significantly when `KL` is removed. In the `Human Preference Alignment` task, `KL` prevents a collapse in visual diversity, where outputs converge to a single style.
*   **Conclusion:** `KL regularization` is not merely an `early stopping` mechanism. A properly tuned `KL` term (e.g., $\beta = 0.04$ for GenEval/Text Rendering, $\beta = 0.01$ for Pickscore) allows `Flow-GRPO` to match the high task rewards of the `KL-free` version while preserving image quality and diversity, though it might require longer training.
    The following figure (Figure 6 from the original paper) visually demonstrates the effect of `KL Regularization`:

    ![Figure 6: Effect of KL Regularization. The KL penalty effectively suppresses reward hacking preventing Quality Degradation (for GenEval and OCR) and Diversity Decline (for PickScore).](images/6.jpg)
    *该图像是一个示意图，展示了KL正则化的效果。左侧的‘Quality Degradation’部分对比了不同模型生成的苹果图像质量，右侧的‘Diversity Decline’部分则展示了不同模型生成的林肯演讲图像多样性。采用KL正则化的图像在质量与多样性上均表现优异。*

The following figure (Figure 12 from the original paper) shows learning curves with and without `KL` for all three tasks:

![Figure 12: Learning Curves with and without KL. KL penalty slows early training yet effectively suppresses reward hacking.](images/12.jpg)
*该图像是图表，展示了在训练步骤中，使用和不使用 KL 的情况下在多个任务中的评估结果。左侧(a)为图像生成的 GenEval 分数，中间(b)为视觉文本渲染的 OCR 准确率，右侧(c)为人类偏好对齐的 PickScore。通过 KL 惩罚能有效抑制奖励黑客行为。*

This further emphasizes that `KL` penalty slows early training but effectively suppresses `reward hacking`, leading to more robust models.

### 6.2.2. Effect of `Denoising Reduction`
The `Denoising Reduction` strategy is crucial for training efficiency.
*   **Observation:** Figure 7(a) shows that reducing `denoising steps` during training from `40` to `10` achieves over a $4\times$ speedup (convergence in terms of GPU time) without impacting the final reward on the GenEval task. Further reduction to `5` steps does not consistently improve speed and can sometimes slow training or make it unstable.
*   **Conclusion:** Using a moderate number of `denoising steps` (e.g., `10`) during training is an effective trade-off, enabling faster convergence without sacrificing final performance at inference (where `40` steps are used). This confirms that `low-quality but informative trajectories` are sufficient for `RL` learning.
    The following figure (Figure 7 from the original paper) illustrates the effect of `Denoising Reduction` on GenEval:

    ![Figure 7: Ablation studies on our critical design choices. (a) Denoising Reduction: Fewer denoising steps accelerate convergence and yield similar performance. (b) Noise Level: Moderate noise level b $a = 0 . 7$ ) maximises OCR accuracy, while too little noise hampers exploration.](images/7.jpg)
    *该图像是图表，展示了去噪减少对GenEval得分和噪声水平消融对OCR准确度的影响。图(a)显示不同去噪步骤在GPU训练时间中的表现，图(b)显示不同噪声水平$a$对OCR准确度的影响，最佳噪声水平为$a = 0.7$。*

The following figure (Figure 9 from the original paper) provides extended `Denoising Reduction` ablations for `Visual Text Rendering` and `Human Preference Alignment`:

![Figure 9: Effect of Denoising Reduction](images/9.jpg)
*该图像是图表，展示了 Flow-GRPO 在视觉文本渲染和人类偏好对齐方面的训练效果。左侧图表显示 OCR 评估准确率随着训练时间的变化，右侧图表呈现 PickScore 的变化趋势。不同步骤数的效果被标记，显示了训练效率的提升。*

These graphs confirm similar trends across tasks: fewer steps ($T=10$) significantly accelerate training while achieving comparable final performance.

### 6.2.3. Effect of `Noise Level` ($a$)
The parameter $a$ in $\sigma_t = a\sqrt{\frac{t}{1-t}}$ controls the level of stochasticity injected into the `SDE`.
*   **Observation:** Figure 7(b) shows that a small $a$ (e.g., `0.1`) limits exploration and slows `reward improvement`. Increasing $a$ up to `0.7` boosts exploration and speeds up `reward gains` (maximizing `OCR accuracy`). Beyond `0.7` (e.g., `1.0`), further increases provide no additional benefit, as exploration is already sufficient.
*   **Conclusion:** A moderate `noise level` is optimal. Too much noise can degrade image quality, leading to zero reward and `failed training`, indicating a balance between exploration and maintaining image coherence is necessary.
    The following figure (Figure 7 from the original paper) illustrates the effect of `Noise Level`:

    ![Figure 7: Ablation studies on our critical design choices. (a) Denoising Reduction: Fewer denoising steps accelerate convergence and yield similar performance. (b) Noise Level: Moderate noise level b $a = 0 . 7$ ) maximises OCR accuracy, while too little noise hampers exploration.](images/7.jpg)
    *该图像是图表，展示了去噪减少对GenEval得分和噪声水平消融对OCR准确度的影响。图(a)显示不同去噪步骤在GPU训练时间中的表现，图(b)显示不同噪声水平$a$对OCR准确度的影响，最佳噪声水平为$a = 0.7$。*

### 6.2.4. Effect of `Group Size` ($G$)
The `group size` $G$ is crucial for `GRPO`'s advantage estimation.
*   **Observation:** Figure 5 shows that reducing `group size` to $G=12$ and $G=6$ led to unstable training and eventual collapse when using `PickScore` as the reward function. $G=24$ remained stable.
*   **Conclusion:** Smaller `group sizes` produce inaccurate `advantage estimates`, increasing variance and leading to `training collapse`. A sufficiently large `group size` (e.g., $G=24$) is necessary for stable and effective `GRPO` training, consistent with findings in other `RL` literature [71, 72].
    The following figure (Figure 5 from the original paper) shows ablation studies on different `Group Size G`:

    ![Figure 5: Ablation Studies on Different Group Size $G$ Higher group size performs better.](images/5.jpg)
    *该图像是图表，展示了不同组大小 $G$ 对 Flow-GRPO 训练步骤的影响。可以看到，组大小为 24 时的评估分数最高，而组大小为 6 时的评估效果明显下降，表明更高的组大小带来了更好的性能。*

### 6.2.5. Generalization Analysis
`Flow-GRPO` demonstrates strong generalization capabilities.
*   **Unseen GenEval Scenarios:** Table 4 shows `Flow-GRPO` generalizes well to `unseen objects` (trained on 60, evaluated on 20 unseen) and `unseen counting` (trained on 2-4 objects, evaluated on 5-6 or 12 objects). For instance, it increases `Overall` accuracy on `unseen objects` from `0.64` to `0.90` and `Counting` accuracy for 5-6 objects from `0.13` to `0.48`.
*   **T2I-CompBench++ [6, 73]:** Table 3 indicates significant gains on `T2I-CompBench++`, a benchmark for open-world compositional `T2I` generation with object classes and relationships substantially different from the GenEval-style training data. For example, `SD3.5-M+Flow-GRPO` improves `2D-Spatial` from `0.2850` to `0.5447`.
*   **Conclusion:** The learned capabilities are not just memorized but generalize to novel compositional challenges, showcasing the model's enhanced reasoning.

    The following are the results from Table 3 of the original paper:

    <table>
    <thead>
    <tr>
    <th>Model</th>
    <th>Color</th>
    <th>Shape</th>
    <th>Texture</th>
    <th>2D-Spatial</th>
    <th>3D-Spatial</th>
    <th>Numeracy</th>
    <th>Non-Spatial</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Janus-Pro-7B [69]</td>
    <td>0.5145</td>
    <td>0.3323</td>
    <td>0.4069</td>
    <td>0.1566</td>
    <td>0.2753</td>
    <td>0.4406</td>
    <td>0.3137</td>
    </tr>
    <tr>
    <td>EMU3 [67]</td>
    <td>0.7913</td>
    <td>0.5846</td>
    <td>0.7422</td>
    <td></td>
    <td>—</td>
    <td></td>
    <td>—</td>
    </tr>
    <tr>
    <td>FLUX.1 Dev [5]</td>
    <td>0.7407</td>
    <td>0.5718</td>
    <td>0.6922</td>
    <td>0.2863</td>
    <td>0.3866</td>
    <td>0.6185</td>
    <td>0.3127</td>
    </tr>
    <tr>
    <td>SD3.5-M [4]</td>
    <td>0.7994</td>
    <td>0.5669</td>
    <td>0.7338</td>
    <td>0.2850</td>
    <td>0.3739</td>
    <td>0.5927</td>
    <td>0.3146</td>
    </tr>
    <tr>
    <td>SD3.5-M+Flow-GRPO</td>
    <td style="color: blue;">0.8379</td>
    <td style="color: blue;">0.6130</td>
    <td>0.7236</td>
    <td style="color: blue;">0.5447</td>
    <td style="color: blue;">0.4471</td>
    <td style="color: blue;">0.6752</td>
    <td style="color: blue;">0.3195</td>
    </tr>
    </tbody>
    </table>

The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="7">Unseen Objects</th>
<th colspan="2">Unseen Counting</th>
</tr>
<tr>
<th>Overall</th>
<th>Single Obj.</th>
<th>Two Obj.</th>
<th>Counting</th>
<th>Colors</th>
<th>Position</th>
<th>Attr. Binding</th>
<th>5-6 Objects</th>
<th>12 Objects</th>
</tr>
</thead>
<tbody>
<tr>
<td>SD3.5-M</td>
<td>0.64</td>
<td>0.96</td>
<td>0.73</td>
<td>0.53</td>
<td>0.87</td>
<td>0.26</td>
<td>0.47</td>
<td>0.13</td>
<td>0.02</td>
</tr>
<tr>
<td>SD3.5-M+Flow-GRPO</td>
<td style="color: blue;">0.90</td>
<td style="color: blue;">1.00</td>
<td style="color: blue;">0.94</td>
<td style="color: blue;">0.86</td>
<td style="color: blue;">0.97</td>
<td style="color: blue;">0.84</td>
<td style="color: blue;">0.77</td>
<td style="color: blue;">0.48</td>
<td style="color: blue;">0.12</td>
</tr>
</tbody>
</table>

### 6.2.6. Comparison with Other Alignment Methods (Extended)
*   **Online vs. Offline:** Figure 8 illustrates `Flow-GRPO`'s superior performance over SFT, Flow-RWR, Flow-DPO, and their online variants on the `Human Preference Alignment` task. The online variants (e.g., Online DPO) generally outperform their offline counterparts, confirming the benefits of `online` interaction.
*   **DDPO Comparison:** `DDPO`, when adapted to `flow matching models`, showed slower `reward increases` and eventually `collapsed` in later stages, whereas `Flow-GRPO` trained stably and improved consistently.
*   **ReFL Comparison:** `Flow-GRPO` also surpassed `ReFL` (which requires `differentiable rewards`), highlighting its robustness and generalizability as it does not impose this constraint.
*   **ORW Comparison:** Table 5 and Table 6 compare `Flow-GRPO` with `ORW`. `Flow-GRPO` consistently achieves higher `PickScore` over training steps (Table 5) and outperforms `ORW` in both `CLIP Score` (proxy for `text-image alignment`) and `Diversity Score` (Table 6). This further solidifies `Flow-GRPO`'s advantage in maintaining diversity while aligning with preferences.

    The following are the results from Table 5 of the original paper:

    <table>
    <thead>
    <tr>
    <th>Method</th>
    <th>Step 0</th>
    <th>Step 240</th>
    <th>Step 480</th>
    <th>Step 720</th>
    <th>Step 960</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>SD3.5-M + ORW</td>
    <td>28.79</td>
    <td>29.05</td>
    <td>29.15</td>
    <td>27.58</td>
    <td>23.05</td>
    </tr>
    <tr>
    <td>SD3.5-M + Flow-GRPO</td>
    <td>28.79</td>
    <td>29.10</td>
    <td>29.17</td>
    <td>29.51</td>
    <td>29.89</td>
    </tr>
    </tbody>
    </table>

The following are the results from Table 6 of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>CLIP Score ↑</th>
<th>Diversity Score ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>SD3.5-M</td>
<td>27.99</td>
<td>0.96</td>
</tr>
<tr>
<td>SD3.5-M + ORW</td>
<td>28.40</td>
<td>0.97</td>
</tr>
<tr>
<td>SD3.5-M + Flow-GRPO</td>
<td style="color: blue;">30.18</td>
<td style="color: blue;">1.02</td>
</tr>
</tbody>
</table>

### 6.2.7. Effect of `Initial Noise`
*   **Observation:** Figure 10 shows that initializing each rollout with different random noise (to increase `exploratory diversity`) consistently achieved higher rewards during training compared to using the same initial noise for all rollouts.
*   **Conclusion:** This supports the importance of diverse exploration during `RL` training for stable and effective learning.
    The following figure (Figure 10 from the original paper) shows the effect of `Initial Noise`:

    ![Figure 10: Effect of Initial Noise](images/10.jpg)
    *该图像是一个图表，展示了在训练步骤与 PickScore 评估之间的关系，比较了使用不同初始噪声和相同初始噪声的 Flow GRPO 方法的效果。随着训练步骤的增加，两条曲线显示出明显的上升趋势。*

### 6.2.8. Additional Results on `FLUX.1-Dev`
*   **Observation:** `Flow-GRPO` applied to `FLUX.1-Dev` (another `flow matching model`) using `PickScore` as reward also showed a steady increase in reward throughout training without noticeable `reward hacking` (Figure 11). Table 7 confirms improvements in `Aesthetic`, `ImageReward`, `PickScore`, and `UnifiedReward` for `FLUX.1-Dev + Flow-GRPO` compared to the base `FLUX.1-Dev`.
*   **Conclusion:** This demonstrates `Flow-GRPO`'s generalizability beyond `SD3.5-M` to other `flow matching model` architectures.
    The following figure (Figure 11 from the original paper) shows additional results on `FLUX.1-Dev`:

    ![Figure 11: Additional Results on FLUX.1-Dev](images/11.jpg)
    *该图像是图表，展示了在 FLUX.1 Dev 数据集上使用 Flow-GRPO 方法的训练步骤与 PickScore 评估的关系。随着训练步骤的增加，PickScore 评估值逐渐上升，最终达到 23.43，明显高于未使用 Flow-GRPO 方法时的 21.94。*

The following are the results from Table 7 of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th>Aesthetic</th>
<th>DeQA</th>
<th>ImageReward</th>
<th>PickScore</th>
<th>UnifiedReward</th>
</tr>
</thead>
<tbody>
<tr>
<td>FLUX.1-Dev</td>
<td>5.71</td>
<td>4.31</td>
<td>0.85</td>
<td>22.62</td>
<td>3.65</td>
</tr>
<tr>
<td>FLUX.1-Dev + Flow-GRPO</td>
<td style="color: blue;">6.02</td>
<td>4.24</td>
<td style="color: blue;">1.32</td>
<td style="color: blue;">23.97</td>
<td style="color: blue;">3.81</td>
</tr>
</tbody>
</table>

### 6.2.9. Training Sample Visualization with `Denoising Reduction`
*   **Observation:** Figure 19 visualizes samples under different inference settings: `ODE` (40 steps), `SDE` (40 steps), `SDE` (10 steps), and `SDE` (5 steps). `ODE` (40) and `SDE` (40) yield visually indistinguishable high-quality images, confirming the `ODE-to-SDE` conversion preserves quality. However, `SDE` (10) and `SDE` (5) steps introduce artifacts like color drift and blur, resulting in lower-quality images.
*   **Conclusion:** Despite the lower quality of samples generated with fewer steps, this `Denoising Reduction` strategy *accelerates optimization* because `Flow-GRPO` relies on *relative preferences*. The model still extracts a useful `reward signal`, while significantly cutting wall-clock time, leading to faster convergence without sacrificing final performance.
    The following figure (Figure 19 from the original paper) visualizes `training samples` under different `inference settings`:

    ![该图像是多张不同风格的欢迎拉斯维加斯的路标插图，展示了各式各样的灯光效果和设计。在夜晚的背景中，每个标志独具特色，体现了拉斯维加斯的独特魅力。](images/19.jpg)
    *该图像是多张不同风格的欢迎拉斯维加斯的路标插图，展示了各式各样的灯光效果和设计。在夜晚的背景中，每个标志独具特色，体现了拉斯维加斯的独特魅力。*

## 6.3. Qualitative Results
Figures 13, 14, 15, 16, 17, and 18 from the appendix provide extensive qualitative comparisons and insights into the model's behavior:
*   **GenEval, OCR, PickScore Rewards:** These figures show that `Flow-GRPO` with `KL regularization` dramatically improves the target capability (e.g., correct object counts, legible text, preferred styles) while maintaining overall image quality. In contrast, removing `KL` often leads to visual degradation or loss of diversity.
*   **Evolution of Evaluation Images:** Figures 16, 17, and 18 illustrate how the generated images for fixed prompts progressively improve and align with task objectives over successive training iterations, showcasing the `online RL` learning process.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper introduces `Flow-GRPO`, a pioneering method that successfully integrates `online policy gradient reinforcement learning (RL)` into `flow matching models` for text-to-image (T2I) generation. The core innovation lies in addressing the fundamental challenges of applying `RL` to these models: their deterministic nature and high sampling cost. `Flow-GRPO` achieves this through two key strategies:
1.  **ODE-to-SDE Conversion:** Transforms the deterministic `Ordinary Differential Equation (ODE)` sampling of `flow matching models` into an equivalent `Stochastic Differential Equation (SDE)` framework. This crucial step introduces the necessary stochasticity for `RL exploration` while rigorously preserving the original model's marginal distributions.
2.  **Denoising Reduction Strategy:** Significantly reduces the number of `denoising steps` during `RL` training (for efficient data collection) while retaining the full number of steps for inference (to ensure high-quality outputs). This strategy drastically improves sampling efficiency and training speed.

    Empirically, `Flow-GRPO` demonstrates state-of-the-art performance across diverse `T2I` tasks. It boosts `SD3.5-M`'s accuracy on the challenging GenEval compositional generation benchmark from $63\%$ to an impressive $95\%$, outperforming even `GPT-4o`. Similarly, visual text rendering accuracy improves from $59\%$ to $92\%$, and substantial gains are achieved in human preference alignment. A critical finding is the effectiveness of `KL regularization` in preventing `reward hacking`, ensuring that performance gains do not come at the expense of overall image quality or diversity. `Flow-GRPO` offers a simple, general, and robust framework for applying `online RL` to `flow-based generative models`, opening new avenues for controllable and aligned image synthesis.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations and propose directions for future research:

1.  **Reward Design:** While `Flow-GRPO` shows promise for video generation, current `reward models` (e.g., object detectors, trackers) are often simple heuristics. More advanced `reward models` are needed to capture complex attributes like physical realism and temporal consistency in videos.
2.  **Balancing Multiple Rewards:** Video generation typically involves optimizing multiple, sometimes conflicting, objectives (e.g., realism, smoothness, coherence). Balancing these competing goals remains a challenge requiring careful tuning.
3.  **Scalability:** Video generation is significantly more resource-intensive than `T2I`. Applying `Flow-GRPO` at scale for video tasks will require more efficient data collection and training pipelines.
4.  **Reward Hacking Prevention:** Although `KL regularization` helps, it can lead to longer training times, and occasional `reward hacking` may still occur for specific prompts. Exploring better, more robust methods for preventing `reward hacking` is an ongoing area of research.

## 7.3. Personal Insights & Critique
This paper presents a highly impactful contribution by successfully integrating `online RL` into `flow matching models`, which represents a significant step towards more controllable and alignable `T2I` generation.

**Innovations and Strengths:**
*   **Elegant Solution to a Core Problem:** The `ODE-to-SDE` conversion is a technically elegant solution to the fundamental incompatibility between deterministic `flow models` and stochastic `RL exploration`. It allows pre-trained, high-quality `flow models` to be fine-tuned with `RL` without extensive architectural changes or full retraining, which is highly practical.
*   **Practical Efficiency:** The `Denoising Reduction` strategy is a brilliant practical innovation. Recognizing that `RL` doesn't always need pristine samples for learning relative preferences dramatically cuts down training costs, making `online RL` feasible for large generative models. This highlights a pragmatic approach to `RL` data efficiency.
*   **Comprehensive Validation:** The extensive experiments across `compositional generation`, `text rendering`, and `human preference alignment` with various baselines and ablation studies (especially on `KL regularization`, `noise level`, `group size`) thoroughly demonstrate the method's effectiveness and robustness. The clear evidence against `reward hacking` (with `KL`) is particularly reassuring.
*   **Generalizability:** The results on `FLUX.1-Dev` and `T2I-CompBench++` showcase the method's potential applicability across different `flow-based architectures` and broader, more complex compositional settings.

**Potential Issues & Areas for Improvement/Further Research:**
*   **Hyperparameter Sensitivity:** As noted by the authors, the `KL regularization` coefficient $\beta$ and `noise level` $a$ are crucial hyperparameters. Finding optimal values can be challenging and task-dependent. While the paper provides guidance, developing adaptive or less sensitive `RL` variants could further improve usability.
*   **Complexity of Reward Models:** While `Flow-GRPO` can utilize non-differentiable `reward models` (a strength), the quality of `RL` fine-tuning is inherently tied to the quality of the `reward signal`. Current `reward models` (even advanced `VLMs`) still have limitations and might not fully capture nuanced human preferences or complex task requirements. Future work might need to focus on jointly improving `reward models` and `RL` algorithms.
*   **Interpretability of `SDE` Conversion:** While mathematically sound, the `SDE` conversion introduces a `score function` term that modifies the `velocity field`. A deeper understanding or visualization of how this modified `velocity field` behaves, especially with different $\sigma_t$ schedules, could offer more insights into the `RL`'s exploration mechanism.
*   **Scaling to Higher Resolutions and Video:** The authors correctly identify scalability to video as a limitation. `Denoising Reduction` helps, but `online RL` on very high-resolution images or videos still faces immense computational hurdles related to memory and processing power. Exploring more sophisticated `experience replay` or `off-policy RL` techniques adapted for generative models might further improve data efficiency.
*   **Interaction with Pre-trained Weights:** The `KL regularization` helps keep the model close to its pre-trained weights. While beneficial for quality preservation, there might be scenarios where more aggressive deviation from the pre-trained `policy` is desired for novel capabilities. Investigating dynamic `KL` weighting or alternative regularization schemes could be interesting.

    Inspiration from this paper includes the realization that `RL`'s power for reasoning and alignment can indeed be unlocked for efficient `ODE-based generative models` with clever theoretical and practical adjustments. The `ODE-to-SDE` conversion paradigm could be a powerful tool for injecting stochasticity into other deterministic processes for `RL` or other applications. The emphasis on carefully managing `reward hacking` through `KL regularization` is a valuable lesson for all `RL` applications in complex domains.