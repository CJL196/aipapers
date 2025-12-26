# 1. Bibliographic Information
## 1.1. Title
Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

## 1.2. Authors
- Boyuan Chen (MIT CSAIL)
- Diego Marti Monso (Technical University of Munich)
- Yilun Du (MIT CSAIL)
- Max Simchowitz (MIT CSAIL)
- Russ Tedrake (MIT CSAIL)
- Vincent Sitzmann (MIT CSAIL)

  The authors are affiliated with the MIT Computer Science and Artificial Intelligence Laboratory (CSAIL) and the Technical University of Munich (TUM). These are world-renowned institutions for robotics, computer vision, and machine learning research. The author list includes prominent researchers in areas like robotics (Russ Tedrake), generative models (Yilun Du, Vincent Sitzmann), and reinforcement learning, indicating a strong cross-disciplinary expertise relevant to the paper's focus.

## 1.3. Journal/Conference
The paper was published on arXiv as a preprint. The specific conference or journal it was submitted to is not listed in this version, but given the topic and authors, it would be a suitable submission for top-tier machine learning or computer vision conferences like NeurIPS, ICML, or CVPR.

## 1.4. Publication Year
The initial version was submitted to arXiv in July 2024.

## 1.5. Abstract
The paper introduces **Diffusion Forcing**, a novel training paradigm for diffusion models. This method trains a model to denoise a sequence of tokens where each token has an independent, randomly assigned noise level. The authors apply this to sequence modeling by training a causal model (like an RNN or masked transformer) to predict future tokens without requiring past tokens to be completely noise-free. This approach successfully merges the advantages of traditional next-token prediction models (e.g., variable-length generation) with those of full-sequence diffusion models (e.g., guidance for sampling). The paper demonstrates several unique capabilities, including stable long-horizon generation of continuous data (like video) past the training horizon, and new sampling/guidance techniques like **Monte Carlo Guidance (MCG)** that significantly improve performance in decision-making and planning. The authors also provide a theoretical proof that their method optimizes a variational lower bound on the likelihood of all possible subsequences.

## 1.6. Original Source Link
- **Original Source Link:** https://arxiv.org/abs/2407.01392
- **PDF Link:** https://arxiv.org/pdf/2407.01392v4.pdf
- **Publication Status:** This is a preprint on arXiv and has not yet been peer-reviewed and published in a conference or journal.

# 2. Executive Summary
## 2.1. Background & Motivation
Probabilistic sequence modeling is fundamental to many areas of AI, from natural language processing to video generation and robotics. Two dominant paradigms have emerged, each with distinct strengths and weaknesses:

1.  **Next-Token Prediction (Autoregressive Models):** These models, trained with `teacher forcing`, predict the very next token in a sequence based on the ground-truth history.
    *   **Strengths:** They are flexible, allowing for variable-length generation, efficient conditioning on past context, and integration with search algorithms like tree search.
    *   **Weaknesses:**
        *   **Error Accumulation:** For continuous data like video or robot states, small prediction errors at each step accumulate, causing the generated sequence to diverge and become unstable, especially over long horizons.
        *   **Lack of Guidance:** The autoregressive structure makes it difficult to guide the entire sequence generation process toward a global objective (e.g., maximizing a future reward), as each token is generated sequentially and irrevocably.

2.  **Full-Sequence Diffusion Models:** These models learn the joint distribution of an entire fixed-length sequence by diffusing all tokens simultaneously with the same noise level.
    *   **Strengths:**
        *   **High-Quality Generation:** They excel at generating high-fidelity, continuous data like images and videos.
        *   **Guidance:** They support `guidance`, a powerful technique to steer the sampling process towards sequences that satisfy certain properties (e.g., high reward in a planning task), making them excellent for long-horizon planning.
    *   **Weaknesses:**
        *   **Fixed Length:** They are typically trained and sampled on fixed-length sequences, lacking the flexibility of autoregressive models.
        *   **Non-Causal:** They are usually implemented with non-causal architectures (like an unmasked Transformer), which does not naturally model the temporal flow of information.

            The core problem is that these two paradigms offer complementary but mutually exclusive benefits. An application might need the flexibility of next-token prediction *and* the guidance capabilities of full-sequence diffusion, but no single framework provided both.

The paper's innovative idea is to re-conceptualize the training process. Instead of treating all tokens in a sequence as either "fully known" (teacher forcing) or "equally noisy" (full-sequence diffusion), **Diffusion Forcing** proposes training a model to handle sequences where **each token can have its own independent noise level**. This simple but powerful idea unifies the two paradigms.

## 2.2. Main Contributions / Findings
The paper's main contributions are:

1.  **Proposal of Diffusion Forcing (DF):** A new training and sampling paradigm for sequence models. It trains a causal model to denoise sequences of tokens where each token has an independent, per-token noise level. This effectively teaches the model to predict any token given a history of other tokens at *any* stage of noisiness.

2.  **Unification of Model Capabilities:** The resulting model, Causal Diffusion Forcing (CDF), combines the key strengths of both autoregressive and full-sequence diffusion models:
    *   **Flexible Generation:** It can generate variable-length sequences, from a single next token to very long rollouts.
    *   **Long-Horizon Guidance:** It supports guidance to optimize global sequence properties, crucial for planning.
    *   **Stability:** It can generate long sequences of continuous data (e.g., video) far beyond the training horizon without diverging, a major failure mode of traditional autoregressive models.

3.  **Novel Sampling and Guidance Schemes:** The unique architecture of Diffusion Forcing enables new, powerful techniques:
    *   **Monte Carlo Guidance (MCG):** A new guidance method for decision-making that averages guidance gradients over multiple sampled future trajectories. This provides a better estimate of expected future rewards and significantly improves planning performance.
    *   **Flexible Sampling Schedules:** The ability to control per-token noise levels during sampling allows for modeling "causal uncertainty" (keeping the far future more uncertain than the near future), which improves guidance.

4.  **Theoretical Justification:** The paper proves that the Diffusion Forcing training objective optimizes a variational lower bound (ELBO) on the likelihood of *all possible subsequences* of the training data. This provides a solid theoretical foundation for the method's empirical success.

5.  **Broad Empirical Validation:** The paper demonstrates the effectiveness of Diffusion Forcing across a diverse set of tasks, including video prediction, model-based planning in reinforcement learning, long-horizon robotic imitation learning, and time-series forecasting, showing its versatility as a general-purpose sequence model.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that learn to create data by reversing a gradual noising process. They consist of two main parts:

*   **Forward Process (Noising):** This is a fixed process where Gaussian noise is incrementally added to a clean data sample $\mathbf{x}^0$ over a series of $K$ timesteps. The data at step $k$, denoted $\mathbf{x}^k$, is a slightly noised version of the data at step `k-1`. The process is defined as a Markov chain:
    \$
    q(\mathbf{x}^k | \mathbf{x}^{k-1}) = \mathcal{N}(\mathbf{x}^k; \sqrt{1 - \beta_k} \mathbf{x}^{k-1}, \beta_k \mathbf{I})
    \$
    where $\{\beta_k\}_{k=1}^K$ is a predefined noise schedule. A key property is that we can sample $\mathbf{x}^k$ directly from $\mathbf{x}^0$ at any step $k$:
    \$
    \mathbf{x}^k = \sqrt{\bar{\alpha}_k} \mathbf{x}^0 + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}
    \$
    where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, $\alpha_k = 1 - \beta_k$, and $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$. After many steps ($k \to K$), $\mathbf{x}^K$ becomes indistinguishable from pure Gaussian noise.

*   **Reverse Process (Denoising):** This is where the learning happens. A neural network, $p_\theta$, is trained to reverse the noising process, step-by-step. It learns to predict the distribution of the slightly less noisy data $\mathbf{x}^{k-1}$ given the noisy data $\mathbf{x}^k$:
    \$
    p_\theta(\mathbf{x}^{k-1} | \mathbf{x}^k) = \mathcal{N}(\mathbf{x}^{k-1}; \boldsymbol{\mu}_\theta(\mathbf{x}^k, k), \gamma_k \mathbf{I})
    \$
    Instead of directly predicting the mean $\boldsymbol{\mu}_\theta$, it is common practice to train the network $\boldsymbol{\epsilon}_\theta(\mathbf{x}^k, k)$ to predict the noise $\boldsymbol{\epsilon}$ that was added to get to $\mathbf{x}^k$. The training objective is a simple mean squared error loss:
    \$
    \mathcal{L}(\theta) = \mathbb{E}_{k, \mathbf{x}^0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_k} \mathbf{x}^0 + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}, k) \|^2 \right]
    \$
    To generate a new sample, one starts with pure noise $\mathbf{x}^K \sim \mathcal{N}(0, \mathbf{I})$ and iteratively applies the learned reverse process to denoise it back to a clean sample $\mathbf{x}^0$.

### 3.1.2. Classifier Guidance
Guidance is a technique used during the sampling phase of a diffusion model to steer the generation towards data with desired attributes. For instance, in `classifier guidance`, we can use a separately trained classifier $c(y|\mathbf{x}^k)$ that predicts a label $y$ from a noisy sample $\mathbf{x}^k$. The idea is to modify the noise prediction of the diffusion model to not only denoise the sample but also to make it more likely to belong to the desired class $y$. The updated noise prediction becomes:
\$
\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}^k, k) = \boldsymbol{\epsilon}_\theta(\mathbf{x}^k, k) - w \sqrt{1 - \bar{\alpha}_k} \nabla_{\mathbf{x}^k} \log c(y|\mathbf{x}^k)
\$
Here, the additional term pushes the sample $\mathbf{x}^k$ in the direction that increases the log-probability of the desired class $y$, with $w$ controlling the strength of the guidance. This allows for powerful, zero-shot control over the generation process.

### 3.1.3. Next-Token Prediction and Teacher Forcing
Next-token prediction is the standard paradigm for training autoregressive sequence models like LSTMs or Transformers. The model is trained to predict the next token $\mathbf{x}_{t+1}$ given the preceding sequence of tokens $\mathbf{x}_{1:t}$.

This is typically done using **Teacher Forcing**. During training, the model is always fed the *ground-truth* sequence of previous tokens to predict the next one. For example, to predict $\mathbf{x}_5$, the model is given the true tokens $(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \mathbf{x}_4)$.

*   **Advantage:** This makes training stable and efficient.
*   **Disadvantage (Exposure Bias):** At inference time, the model must generate tokens one by one and feed its own predictions back as input to generate the next token. If it makes a small error, that error can propagate and compound, leading the model into states it never saw during training, often causing the generation to degrade or "diverge." This is a major issue for continuous data.

### 3.1.4. Bayesian Filtering and Hidden Markov Models (HMM)
A Hidden Markov Model (HMM) describes a system with unobserved (latent) states $\mathbf{z}_t$ that evolve over time and produce observations $\mathbf{x}_t$. The core components are:
*   A **transition model** $p(\mathbf{z}_{t+1} | \mathbf{z}_t)$ that describes how the latent state changes.
*   An **observation model** $p(\mathbf{x}_t | \mathbf{z}_t)$ that describes how an observation is generated from the current latent state.

    **Bayesian Filtering** is a recursive algorithm for estimating the current latent state $\mathbf{z}_t$ given all observations up to that point, $\mathbf{x}_{1:t}$. It involves two steps:
1.  **Predict (Prior):** Use the transition model to predict the next state's distribution based on the previous state's estimate: $p(\mathbf{z}_{t+1} | \mathbf{x}_{1:t}) = \int p(\mathbf{z}_{t+1} | \mathbf{z}_t) p(\mathbf{z}_t | \mathbf{x}_{1:t}) d\mathbf{z}_t$.
2.  **Update (Posterior):** When a new observation $\mathbf{x}_{t+1}$ arrives, update the state estimate using Bayes' rule: $p(\mathbf{z}_{t+1} | \mathbf{x}_{1:t+1}) \propto p(\mathbf{x}_{t+1} | \mathbf{z}_{t+1}) p(\mathbf{z}_{t+1} | \mathbf{x}_{1:t})$.

    The paper frames its Causal Diffusion Forcing model within this context, where the RNN's hidden state acts as the latent state $\mathbf{z}_t$, summarizing the history.

## 3.2. Previous Works
The paper positions itself relative to two main lines of work: next-token prediction and full-sequence diffusion.

*   **Next-Token Prediction:** This includes classic models like LSTMs [33] and modern Transformers [6] used in language modeling. For time-series, works like `DeepAR` [55] use autoregressive RNNs for probabilistic forecasting. A key limitation addressed by the current paper is the instability of these models on continuous data and their inability to use long-horizon guidance.

*   **Full-Sequence Diffusion:** This approach has been successful in video generation (`Video Diffusion Models` [32]) and planning (`Diffuser` [37]). `Diffuser` models entire trajectories of `(state, action, reward)` tuples as a single sequence and uses guidance to plan high-reward trajectories. However, it's limited to fixed-horizon planning and uses a non-causal architecture, which can lead to inconsistencies between planned actions and states.

*   **Hybrid/Causal Diffusion Sequence Models:** The most similar prior work is **AR-Diffusion** [66]. It also uses a causal architecture for sequence diffusion but with a critical difference: the noise level for each token is *linearly dependent* on its position in the sequence (e.g., earlier tokens are less noisy). `Diffusion Forcing`'s key innovation is to make the per-token noise levels **independent and random**, which unlocks greater flexibility, including the ability to stabilize generation and condition on arbitrarily corrupted observations without retraining. Another related work, `Rolling Diffusion` [52], also uses a structured noise schedule but shares the same limitations as AR-Diffusion.

    The following diagram from the paper (Figure 2) perfectly illustrates the differences between these paradigms.

    ![Figure 2: Method Overview. Diffusion Forcing trains causal sequence neural networks (such as an RNN or a masked transformer) to denoise flexible-length sequences where each frame of the sequence can have a different noise level. In contrast, next-token prediction models, common in language modeling, are trained to predict a single next token from a ground-truth sequence (teacher forcing \[65\]), and full-sequence diffusion, common in video generation, train non-causal architectures to denoise all frames in a sequence at once with the same noise level. Diffusion Forcing thus interleaves the time axis of the sequence and the noise axis of diffusion, unifying strengths of both alternatives and enabling completely new capabilities (see Secs. 3.2,3.4).](images/2.jpg)
    *该图像是示意图，展示了 Diffusion Forcing、Teacher Forcing 和 Full-Seq. Diffusion 三种不同的序列生成机制。上方部分为训练过程，显示了不同噪声水平下的序列生成；下方为采样过程，展示了各机制的生成流程与噪声添加方式。*

## 3.3. Technological Evolution
The field of sequence modeling has evolved significantly:
1.  **RNNs/LSTMs:** Early dominant models for sequential data, good at capturing short-term dependencies but struggled with long-range ones.
2.  **Transformers:** Revolutionized the field with the `self-attention` mechanism, excelling at capturing long-range dependencies, leading to the rise of Large Language Models (LLMs). Both were primarily trained autoregressively with teacher forcing.
3.  **Diffusion Models:** Emerged as state-of-the-art in generative modeling for images and later video, offering high-quality synthesis and controllability via guidance. Their application to sequences was initially non-causal and for fixed lengths.
4.  **Causal Sequence Diffusion:** Recent works like AR-Diffusion began exploring how to make diffusion models for sequences causal and autoregressive.

    `Diffusion Forcing` represents the next step in this evolution, creating a unified framework that inherits the best properties of both autoregressive models (flexibility, causality) and diffusion models (quality, guidance), while overcoming their respective limitations.

## 3.4. Differentiation Analysis
Compared to the main methods, `Diffusion Forcing`'s core innovations are:

| Feature | Next-Token Prediction | Full-Sequence Diffusion | AR-Diffusion [66] | **Diffusion Forcing (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **Training Noise** | Predicts clean token from clean history (no noise). | All tokens have the **same** noise level. | Noise level is **dependent** on token position (linear schedule). | Each token has an **independent, random** noise level. |
| **Architecture** | Causal | Non-Causal | Causal | Causal |
| **Guidance** | No multi-step guidance | Yes | Not explored, limited potential | Yes, enables novel **Monte Carlo Guidance** |
| **Generation Length**| Variable | Fixed | Variable | Variable |
| **Stability (Continuous Data)** | Poor (diverges) | Good | Improved | **Excellent (stable rollouts)** |
| **Flexibility at Sampling** | Low (fixed autoregressive) | Low (fixed full-sequence) | Low (fixed schedule) | **High (arbitrary noise schedules)** |

The crucial differentiator is the **independent per-token noise** during training. This forces the model to learn a much more general conditional distribution, making it robust enough to handle a wide variety of conditioning scenarios at inference time, from standard autoregression to complex planning with guidance.

# 4. Methodology
## 4.1. Principles
The core idea of `Diffusion Forcing` is to treat the noising process in diffusion as a form of **partial masking**.
*   A token with zero noise ($\mathbf{x}_t^0$) is completely **unmasked** (fully visible).
*   A token with maximum noise ($\mathbf{x}_t^K$) is completely **masked** (pure noise, no information).
*   A token with intermediate noise ($\mathbf{x}_t^k$ for $0 < k < K$) is **partially masked**.

    Traditional sequence modeling methods operate at the extremes. `Teacher forcing` trains a model to predict a fully masked token from a history of fully unmasked tokens. `Full-sequence diffusion` trains a model to denoise a sequence where all tokens are equally partially masked.

`Diffusion Forcing` generalizes this by forcing the model to learn to denoise *any* collection of variably noised tokens. By training on sequences where each token $\mathbf{x}_t$ has an independently and randomly sampled noise level $k_t$, the model learns to handle any combination of masked, unmasked, and partially masked tokens. When applied with a causal architecture, this allows the model to predict future tokens conditioned on a past that isn't perfectly clean, which is the key to its stability and flexibility.

## 4.2. Core Methodology In-depth (Layer by Layer)
The paper implements `Diffusion Forcing` for time-series data with a causal architecture, which they name **Causal Diffusion Forcing (CDF)**. The implementation uses a Recurrent Neural Network (RNN) for simplicity, though it can be extended to Transformers.

### 4.2.1. Model Architecture
The model is a recurrent unit that processes a sequence of tokens one by one. At each time step $t$, the model's state is defined by a latent variable $\mathbf{z}_t$, which summarizes the history up to that point. The model performs two functions:
1.  **Latent State Update:** It updates its latent state based on the previous latent state $\mathbf{z}_{t-1}$ and the current (noisy) observation $\mathbf{x}_t^{k_t}$. This is modeled as a recurrent dynamic:
    \$
    \mathbf{z}_t \sim p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)
    \$
    This step is analogous to the posterior update in a Bayes filter. When $k_t=K$ (pure noise), it models the prior distribution $p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1})$.
2.  **Noise Prediction:** Given the latent state $\mathbf{z}_{t-1}$ and the noisy token $\mathbf{x}_t^{k_t}$, the model predicts the noise $\boldsymbol{\epsilon}_t$ that was added to the original clean token $\mathbf{x}_t^0$. The prediction function is denoted $\boldsymbol{\epsilon}_\theta(\mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)$.

### 4.2.2. Training Process
The training process is detailed in Algorithm 1. For each training iteration:

1.  **Sample Data:** A trajectory of clean observations $(\mathbf{x}_1, \dots, \mathbf{x}_T)$ is sampled from the training dataset.
2.  **Sample Noise Levels:** For each token $\mathbf{x}_t$ in the sequence, an independent noise level $k_t$ is uniformly sampled from $\{0, 1, \dots, K\}$.
3.  **Apply Forward Diffusion:** Each clean token $\mathbf{x}_t$ is noised to its corresponding level $k_t$ using the standard diffusion forward process, producing a sequence of noisy tokens $(\mathbf{x}_1^{k_1}, \dots, \mathbf{x}_T^{k_T})$. The original noise used for each token, $\boldsymbol{\epsilon}_t$, is stored as the target.
4.  **Denoise and Compute Loss:** The model processes the noisy sequence autoregressively. At each step $t$:
    *   The model predicts the noise `\hat{\boldsymbol{\epsilon}}_t = \boldsymbol{\epsilon}_\theta(\mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t)`.
    *   The latent state is updated to get $\mathbf{z}_t$.
5.  **Minimize Loss:** The model's parameters $\theta$ are updated by minimizing the sum of mean squared errors between the predicted noise and the true noise for all tokens in the sequence. The loss function is:
    \$
    \mathcal{L}(\theta) = \underset{\substack{k_{1:T}, \mathbf{x}_{1:T}, \boldsymbol{\epsilon}_{1:T}}}{\mathbb{E}} \sum_{t=1}^{T} \left\| \boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta \big( \mathbf{z}_{t-1}, \mathbf{x}_t^{k_t}, k_t \big) \right\|^2
    \$
    where the expectation is over training data, uniformly sampled noise levels $k_{1:T}$, and the corresponding noise samples $\boldsymbol{\epsilon}_{1:T}$. The latent $\mathbf{z}_{t-1}$ is computed recurrently based on the history $(\mathbf{x}_s^{k_s})_{s<t}$.

### 4.2.3. Theoretical Justification (Theorem 3.1)
The paper provides a formal proof in Appendix A that this training objective is not just an ad-hoc heuristic but optimizes a well-defined probabilistic objective.
**Theorem 3.1 (Informal):** The `Diffusion Forcing` training procedure optimizes a reweighted Evidence Lower Bound (ELBO) on the log-likelihood of sequences of tokens drawn from the true data distribution. The expectation is averaged over all possible combinations of noise levels. Furthermore, under ideal conditions (a fully expressive model), optimizing this single objective simultaneously maximizes a lower bound on the likelihood for *all possible sequences of noise levels*.

This is a powerful result. It means the model isn't just learning an "average" behavior. It is learning to model the conditional distribution for any specific pattern of noisy history, including cases like:
*   $k_{1:t-1}=0, k_t=K$: Standard next-token prediction.
*   $k_1, k_2, \dots, k_T$ are all equal: Standard full-sequence diffusion.
*   Some $k_t=K$: Conditioning on a history with missing tokens.
*   Some $k_t > 0$: Conditioning on a corrupted/noisy history.

### 4.2.4. Sampling Process
The flexibility of `Diffusion Forcing` truly shines during sampling, as described in Algorithm 2. Unlike standard diffusion, which has a single 1D noise schedule, `Diffusion Forcing` sampling is defined by a 2D **scheduling matrix** $\mathcal{K}$ of size $M \times T$, where $M$ is the number of denoising steps and $T$ is the sequence length. $\mathcal{K}_{m,t}$ specifies the target noise level for token $t$ at denoising step $m$.

The general process is as follows:
1.  **Initialize:** Start with a sequence of pure noise tokens $\mathbf{x}_{1:T}$, all at noise level $K$.
2.  **Iterate and Denoise:** Iterate through the rows of the scheduling matrix $\mathcal{K}$ from $m=M-1$ down to `0`. In each row $m$:
    *   Iterate through the tokens from $t=1$ to $T$.
    *   For each token $\mathbf{x}_t$, perform one step of denoising to move it from its current noise level to the target level $\mathcal{K}_{m,t}$. This is done using the standard reverse diffusion update rule, conditioned on the latent state $\mathbf{z}_{t-1}$ from the previous token.
    *   Update the latent state to $\mathbf{z}_t$ before moving to the next token.
3.  **Apply Guidance (Optional):** After each full row of denoising (i.e., for each step $m$), a guidance gradient can be applied to the partially denoised sequence $\mathbf{x}_{1:T}$ to steer it towards a desired objective.
4.  **Finalize:** After iterating through all rows, the sequence at row $m=0$ is fully denoised (since $\mathcal{K}_{0,t}=0$ for all $t$).

    By designing different scheduling matrices $\mathcal{K}$, one can achieve a wide range of behaviors without retraining the model. For example, the "zig-zag" schedule (Figure 2) denoises earlier tokens faster than later ones, embodying causal uncertainty.

### 4.2.5. New Capabilities in Detail
*   **Stabilizing Autoregressive Generation:** For long video rollouts, instead of feeding the perfectly clean (but potentially slightly erroneous) generated frame $\mathbf{x}_t^0$ to predict $\mathbf{x}_{t+1}$, the model is conditioned on a slightly re-noised version $\mathbf{x}_t^{k_{\text{small}}}$. Because the model was trained via `Diffusion Forcing` to handle noisy inputs, it remains in-distribution and avoids error accumulation.

*   **Monte Carlo Guidance (MCG):** In decision-making, the goal is to choose an action that maximizes the *expected* future reward. Standard guidance methods guide based on the reward of a single sampled future trajectory. `Diffusion Forcing`'s causal structure and ability to model uncertainty allows for a better approach. At a given step, to guide the current action $\mathbf{a}_t$, one can:
    1.  Keep the current state partially noisy.
    2.  Sample *multiple* possible future trajectories from that point.
    3.  Calculate the guidance gradient for each future sample.
    4.  **Average** these gradients to get a low-variance estimate of the gradient of the *expected* future reward.
    5.  Apply this averaged gradient to guide the generation of $\mathbf{a}_t$.
        This is analogous to shooting methods in optimal control and is uniquely enabled by CDF's ability to represent a distribution over futures.

# 5. Experimental Setup
## 5.1. Datasets
The paper validates `Diffusion Forcing` across a diverse range of sequence modeling tasks.

*   **Video Prediction:**
    *   **Minecraft:** First-person videos of a player navigating a Minecraft world. The data consists of random walks.
    *   **DMLab:** First-person videos of an agent navigating 3D maze environments from DeepMind Lab.
    *   *Reasoning:* These datasets feature continuous, high-dimensional data where autoregressive models typically fail due to error accumulation. They are ideal for testing the long-horizon stability of the model.

*   **Offline Reinforcement Learning (Planning):**
    *   **D4RL Maze2D:** A standard benchmark for offline RL. The task is to navigate a 2D point mass to a goal in different mazes (`medium`, `large`, `u-maze`) with sparse rewards. The dataset consists of suboptimal, random walk trajectories.
    *   *Reasoning:* These are challenging long-horizon planning tasks with sparse rewards, where guidance is essential for success. They are perfect for evaluating the benefits of `Monte Carlo Guidance` and flexible horizon planning.

*   **Robotic Imitation Learning:**
    *   **Real Robot Fruit Swapping:** A long-horizon manipulation task where a Franka robot arm must swap the positions of an apple and an orange using a third empty slot. The initial fruit positions are randomized.
    *   *Reasoning:* This task is non-Markovian; the correct action depends on the memory of the initial state, which standard imitation learning policies lack. It tests the model's ability to use its latent state as memory. The experiment also tests robustness to visual distractors.

*   **Compositional Generation:**
    *   **2D Cross Trajectories:** A synthetic dataset of 2D trajectories that form a cross shape, starting from one corner and moving to the opposite.
    *   *Reasoning:* This simple dataset is used to demonstrate how different sampling schemes can either replicate the training distribution (full memory) or compose sub-trajectories to create novel behaviors (no memory MPC).

*   **Time Series Forecasting:**
    *   **GluonTS Datasets:** A collection of real-world multivariate time series datasets including `Exchange-Rate`, `Solar-Energy`, `Electricity`, `Traffic`, `Taxi`, and `Wikipedia`. They vary in dimension, domain, and frequency.
    *   *Reasoning:* This benchmark tests `Diffusion Forcing`'s performance as a general-purpose probabilistic sequence model against specialized state-of-the-art methods.

        The following figure shows an example of the video prediction task.

        ![Figure 3: Video Generation. Among tested methods, Diffusion Forcing generations are uniquely temporally consistent and do not diverge even when rolling out well past the training horizon. Please see the project website for video results.](images/4.jpg)
        *该图像是一个示意图，展示了不同方法生成的图像序列，包括 DMLab 和 Minecraft 的输入及各自生成的图像。图中显示了 Diffusion Forcing 与其他生成方法的结果对比，强调其在时间一致性上的表现。*

## 5.2. Evaluation Metrics
*   **Video Prediction & Robotics:** Evaluation is primarily **qualitative**. The key metric is temporal consistency and whether the generated sequences diverge or remain stable and realistic over long horizons. For the robotics task, **success rate** (%) is used.

*   **Planning (D4RL):** The metric is the standard **normalized episode reward**. Higher is better.

*   **Time Series Forecasting:** The primary metric is the **Continuous Ranked Probability Score (CRPS)**, summed over the feature dimensions.

    1.  **Conceptual Definition:** CRPS is a proper scoring rule that measures the compatibility between a predicted probability distribution and an observed outcome. It generalizes the mean absolute error to the case of probabilistic forecasts. A lower CRPS indicates that the predicted distribution is a better match for the ground truth observation. The $\text{CRPS}_{\text{sum}}$ aggregates this score across all dimensions of the multivariate time series.

    2.  **Mathematical Formula:** The CRPS for a single predictive cumulative distribution function (CDF) $F$ and a single observation $x$ is:
        \$
        \mathrm{CRPS}(F, x) = \int_{-\infty}^{\infty} (F(z) - \mathbb{I}\{x \leq z\})^2 dz
        \$

    3.  **Symbol Explanation:**
        *   `F(z)`: The predicted CDF, which gives the probability that the outcome will be less than or equal to $z$.
        *   $x$: The single, observed ground-truth value.
        *   $\mathbb{I}\{x \leq z\}$: The indicator function, which is 1 if the condition $x \leq z$ is true, and 0 otherwise. This represents the CDF of a perfect forecast concentrated at the true value $x$.
        *   The integral measures the squared difference between the predicted and ideal CDFs over all possible outcomes.

## 5.3. Baselines
*   **Video Prediction:**
    *   `Next-frame diffusion`: An autoregressive model trained with teacher forcing to predict the next frame using diffusion.
    *   `Causal full-sequence diffusion`: A diffusion model with a causal architecture trained to denoise the entire sequence with a single shared noise level.

*   **Planning (D4RL):**
    *   `Diffuser`: The state-of-the-art diffusion-based planner.
    *   `CQL` (Conservative Q-Learning) and `IQL` (Implicit Q-Learning): State-of-the-art value-based offline RL methods.
    *   `MPPI`: A classic model-predictive control algorithm.

*   **Time Series Forecasting:**
    *   `TimeGrad`: An autoregressive diffusion model for time series.
    *   `ScoreGrad`: A continuous energy-based generative model.
    *   `Transformer-MAF`: A transformer-based model with normalizing flows.
    *   Classical methods like `VAR`, `DeepAR`, and `GP-Copula`.

*   **Robotics:**
    *   `Diffusion Policy`: A state-of-the-art imitation learning algorithm without explicit memory.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### 6.1.1. Video Prediction: Stabilizing Long Rollouts
The results in Figure 3 clearly show the superiority of `Diffusion Forcing` (CDF).
*   **CDF (`Ours`)** generates temporally consistent and stable video sequences, even when rolled out for 1000 frames, far beyond the training horizon (e.g., 72 frames). The agent is shown moving smoothly through the environment.
*   **`Teacher Forcing`** (next-frame prediction) diverges very quickly. The generated frames become noisy and nonsensical as errors accumulate.
*   **`Full-Sequence Diffusion`** (with a causal mask) also struggles. While it doesn't diverge into pure noise, it suffers from severe discontinuity, with the scene "jumping" dramatically between frames. This shows that simply adding a causal mask to a full-sequence diffusion model is not enough.

    This experiment strongly validates that `Diffusion Forcing`'s unique training scheme is essential for achieving stable long-horizon generation of continuous data.

### 6.1.2. Diffusion Planning: The Power of MCG and Causality
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th>Environment</th>
<th>MPPI</th>
<th>CQL</th>
<th>IQL</th>
<th>Diffuser*</th>
<th>Diffuser w/ diffused action</th>
<th>Ours wo/ MCG</th>
<th>Ours</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8" align="center"><strong>Single-task</strong></td>
</tr>
<tr>
<td>Maze2D U-Maze</td>
<td>33.2</td>
<td>5.7</td>
<td>47.4</td>
<td>113.9 ± 3.1</td>
<td>6.3 ± 2.1</td>
<td>110.1 ± 3.9</td>
<td><strong>116.7 ± 2.0</strong></td>
</tr>
<tr>
<td>Maze2D Medium</td>
<td>10.2</td>
<td>5.0</td>
<td>34.9</td>
<td>121.5 ± 2.7</td>
<td>13.5 ± 2.3</td>
<td>136.1 ± 10.2</td>
<td><strong>149.4 ± 7.5</strong></td>
</tr>
<tr>
<td>Maze2D Large</td>
<td>5.1</td>
<td>12.5</td>
<td>58.6</td>
<td>123.0 ± 6.4</td>
<td>6.3 ± 2.1</td>
<td>142.8 ± 5.6</td>
<td><strong>159.0 ± 2.7</strong></td>
</tr>
<tr>
<td>Average</td>
<td>16.2</td>
<td>7.7</td>
<td>47.0</td>
<td>119.5</td>
<td>8.7</td>
<td>129.67</td>
<td><strong>141.7</strong></td>
</tr>
<tr>
<td colspan="8" align="center"><strong>Multi-task</strong></td>
</tr>
<tr>
<td>Multi2D U-Maze</td>
<td>41.2</td>
<td>-</td>
<td>24.8</td>
<td>128.9 ± 1.8</td>
<td>32.8 ± 1.7</td>
<td>107.7 ± 4.9</td>
<td><strong>119.1 ± 4.0</strong></td>
</tr>
<tr>
<td>Multi2D Medium</td>
<td>15.4</td>
<td>-</td>
<td>12.1</td>
<td>127.2 ± 3.4</td>
<td>22.0 ± 2.7</td>
<td>145.6 ± 6.5</td>
<td><strong>152.3 ± 9.9</strong></td>
</tr>
<tr>
<td>Multi2D Large</td>
<td>8.0</td>
<td>-</td>
<td>13.9</td>
<td>132.1 ± 5.8</td>
<td>6.9 ± 1.7</td>
<td>129.8 ± 1.5</td>
<td><strong>167.1 ± 2.7</strong></td>
</tr>
<tr>
<td>Average</td>
<td>21.5</td>
<td>-</td>
<td>16.9</td>
<td>129.4</td>
<td>20.6</td>
<td>127.7</td>
<td><strong>146.2</strong></td>
</tr>
</tbody>
</table>

*Note: $Diffuser*$ refers to the original Diffuser implementation which uses a hand-crafted PD controller, not the generated actions.*

Key takeaways from this table:
*   **CDF Outperforms All Baselines:** `Diffusion Forcing` (`Ours`) achieves the highest scores across all single-task and multi-task environments, significantly outperforming prior SOTA methods like `Diffuser`, `CQL`, and `IQL`.
*   **Benefit of Monte Carlo Guidance (MCG):** Comparing `Ours` with `Ours wo/ MCG` shows a consistent and significant performance gain from using MCG. This confirms that guiding based on the expected future reward is more effective than guiding on a single sample.
*   **Benefit of Causality:** The `Diffuser` model's performance collapses when using its own generated actions (`Diffuser w/ diffused action`) instead of a separate controller. This highlights a critical flaw in non-causal sequence models for planning: the generated actions and states are not self-consistent. In contrast, `Diffusion Forcing`'s causal structure ensures that its generated actions are executable and lead to the predicted states, removing the need for hand-crafted controllers.

### 6.1.3. Robotics and Compositionality
*   **Imitation Learning with Memory:** In the fruit-swapping task, `Diffusion Forcing` achieved an **80% success rate**. The baseline `Diffusion Policy`, which lacks memory, **failed completely (0% success)**. This demonstrates that CDF's recurrent latent state effectively functions as the memory needed to solve long-horizon, non-Markovian tasks. Furthermore, when visual distractors were added, CDF's success rate only dropped to 76%, showcasing its robustness when prompted to treat observations as noisy.
*   **Compositional Generation:** As shown in Figure 7 from the paper, by changing the sampling scheme, CDF can either replicate the full "cross" shape from the training data (by using full memory) or compose segments of the cross to form novel "V" shapes (by using memoryless MPC-style sampling). This demonstrates a new level of controllable generation.

    ![Figure 7: Given a dataset of trajectories (a), Diffusion Forcing models the joint distribution of all subsequences of arbitrary length. At sampling time, we can sample from the trajectory distribution by sampling Diffusion Forcing with full horizon (b) or recover Markovian dynamics by disregarding previous states (c).](images/8.jpg)
    *该图像是示意图，展示了数据集（a）、具有记忆的模型（b）以及不具有记忆的模型（c）在生成轨迹时的表现。每个子图表现了不同的轨迹生成方式，展示了Diffusion Forcing在处理时间序列数据时的效果。*

### 6.1.4. Time Series Forecasting
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>Exchange</th>
<th>Solar</th>
<th>Electricity</th>
<th>Traffic</th>
<th>Taxi</th>
<th>Wikipedia</th>
</tr>
</thead>
<tbody>
<tr>
<td>VES [36]</td>
<td>0.005 ± 0.000</td>
<td>0.900 ± 0.003</td>
<td>0.880 ± 0.004</td>
<td>0.350 ± 0.002</td>
<td></td>
<td></td>
</tr>
<tr>
<td>VAR [45]</td>
<td>0.005 ± 0.000</td>
<td>0.830 ± 0.006</td>
<td>0.039 ± 0.001</td>
<td>0.290 ± 0.001</td>
<td></td>
<td></td>
</tr>
<tr>
<td>VAR-Lasso [45]</td>
<td>0.012 ± 0.000</td>
<td>0.510 ± 0.006</td>
<td>0.025 ± 0.000</td>
<td>0.150 ± 0.002</td>
<td></td>
<td>3.100 ± 0.004</td>
</tr>
<tr>
<td>GARCH [62]</td>
<td>0.023 ± 0.000</td>
<td>0.880 ± 0.002</td>
<td>0.190 ± 0.001</td>
<td>0.370 ± 0.001</td>
<td></td>
<td></td>
</tr>
<tr>
<td>DeepAR [55]</td>
<td></td>
<td>0.336 ± 0.014</td>
<td>0.023 ± 0.001</td>
<td>0.055 ± 0.003</td>
<td></td>
<td>0.127 ± 0.042</td>
</tr>
<tr>
<td>LSTM-Copula [54]</td>
<td>0.007 ± 0.000</td>
<td>0.319 ± 0.011</td>
<td>0.064 ± 0.008</td>
<td>0.103 ± 0.006</td>
<td>0.326 ± 0.007</td>
<td>0.241 ± 0.033</td>
</tr>
<tr>
<td>GP-Copula [54]</td>
<td>0.007 ± 0.000</td>
<td>0.337 ± 0.024</td>
<td>0.025 ± 0.002</td>
<td>0.078 ± 0.002</td>
<td>0.208 ± 0.183</td>
<td>0.086 ± 0.004</td>
</tr>
<tr>
<td>KVAE [41]</td>
<td>0.014 ± 0.002</td>
<td>0.340 ± 0.025</td>
<td>0.051 ± 0.019</td>
<td>0.100 ± 0.005</td>
<td></td>
<td>0.095 ± 0.012</td>
</tr>
<tr>
<td>NKF [14]</td>
<td></td>
<td>0.320 ± 0.020</td>
<td>0.016 ± 0.001</td>
<td>0.100 ± 0.002</td>
<td></td>
<td>0.071 ± 0.002</td>
</tr>
<tr>
<td>Transformer-MAF [51]</td>
<td>0.005 ± 0.003</td>
<td>0.301 ± 0.014</td>
<td>0.021 ± 0.000</td>
<td>0.056 ± 0.001</td>
<td>0.179 ± 0.002</td>
<td>0.063 ± 0.003</td>
</tr>
<tr>
<td>TimeGrad [50]</td>
<td>0.006 ± 0.001</td>
<td>0.287 ± 0.020</td>
<td>0.021 ± 0.001</td>
<td>0.044 ± 0.006</td>
<td>0.114 ± 0.020</td>
<td>0.049 ± 0.002</td>
</tr>
<tr>
<td>ScoreGrad sub-VP SDE [68]</td>
<td>0.006 ± 0.001</td>
<td><strong>0.256 ± 0.015</strong></td>
<td><strong>0.019 ± 0.001</strong></td>
<td>0.041 ± 0.004</td>
<td>0.101 ± 0.004</td>
<td><strong>0.043 ± 0.002</strong></td>
</tr>
<tr>
<td>Ours</td>
<td><strong>0.003 ± 0.001</strong></td>
<td>0.289 ± 0.002</td>
<td>0.023 ± 0.001</td>
<td><strong>0.040 ± 0.004</strong></td>
<td><strong>0.075 ± 0.002</strong></td>
<td>0.085 ± 0.007</td>
</tr>
</tbody>
</table>

*Lower $\text{CRPS}_{\text{sum}}$ is better.*

The results show that `Diffusion Forcing` is highly competitive as a general-purpose sequence model. It achieves state-of-the-art or near-state-of-the-art performance on several datasets, outperforming many specialized models. This demonstrates that the novel training objective does not harm its performance on standard forecasting tasks and can even be beneficial.

## 6.2. Ablation Studies / Parameter Analysis
The primary ablation study is the analysis of `Monte Carlo Guidance` in Table 1. The marked performance drop when removing MCG (`Ours` vs. `Ours wo/ MCG`) directly validates its contribution. The paper also implicitly ablates the core idea of `Diffusion Forcing` by comparing it to `Teacher Forcing` and `Causal Full-Sequence Diffusion` in the video experiments, showing that both alternatives fail where CDF succeeds.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces `Diffusion Forcing`, a novel and powerful training paradigm that unifies autoregressive next-token prediction and full-sequence diffusion. By training a causal model to denoise sequences with independent, per-token noise levels, the resulting Causal Diffusion Forcing (CDF) model inherits the flexibility of the former and the guidance-based controllability of the latter. The paper demonstrates that this approach not only solves long-standing issues like error accumulation in continuous sequence generation but also unlocks entirely new capabilities. Chief among these is Monte Carlo Guidance (MCG), a more effective planning strategy for decision-making. Backed by both strong theoretical justification and comprehensive empirical results across video, planning, robotics, and time-series, `Diffusion Forcing` establishes itself as a highly effective and versatile framework for probabilistic sequence modeling.

## 7.2. Limitations & Future Work
The authors acknowledge a few limitations and point to future directions:
*   **Architectural Scaling:** The current implementation is primarily based on an RNN for efficiency. Applying `Diffusion Forcing` to larger, more complex transformer architectures is a natural next step, which could further improve performance, especially on internet-scale datasets.
*   **Broader Applications:** While the paper explores time series, future work could apply `Diffusion Forcing` to other domains, such as language or audio generation. The core principle of independent noising is domain-agnostic.
*   **Scaling to Large Datasets:** The experiments are conducted on standard academic benchmarks. The scaling properties of `Diffusion Forcing` on massive, unlabeled datasets (e.g., pre-training on all of YouTube) remain to be explored.

## 7.3. Personal Insights & Critique
`Diffusion Forcing` is an elegant and powerful idea. Its main strength lies in its conceptual simplicity and the breadth of capabilities it unlocks.

*   **The "Noising as Partial Masking" Analogy:** This is a brilliant insight that provides a unified lens through which to view different sequence training paradigms. It recasts the problem from a discrete choice between methods into a continuous space of "noisiness," which the model learns to navigate. This is a significant conceptual contribution.

*   **Practicality and Flexibility:** The true power of the method is demonstrated at inference time. The ability to use arbitrary `scheduling matrices` ($\mathcal{K}$) to elicit different behaviors (e.g., causal uncertainty, compositional generation) from a single trained model is incredibly powerful. It shifts complexity from model architecture to the sampling process, which is much more flexible.

*   **Monte Carlo Guidance is a Key Innovation:** MCG is a standout contribution for sequential decision-making. It provides a principled way to bridge the gap between single-sample planning and policies that account for future uncertainty. The fact that it arises naturally from the CDF framework highlights the synergy between the architecture and the application.

*   **Potential Issues/Critique:**
    *   **Training Complexity:** Training with independent noise levels for every token in a batch seems computationally more demanding than standard full-sequence diffusion (where all tokens share a noise level) or teacher forcing. The paper mentions `Fused SNR reweighting` to accelerate training, but a deeper analysis of the computational trade-offs would be valuable, especially as models and sequences scale up.
    *   **Optimal Schedule Design:** The paper shows the power of flexible sampling schedules, but how to *design* the optimal scheduling matrix $\mathcal{K}$ for a given task is an open question. It currently appears to be a manually designed hyperparameter, and automating or learning this schedule could be a fruitful area for future research.

        Overall, `Diffusion Forcing` feels like a foundational contribution that could have a lasting impact on how we approach generative sequence modeling, particularly at the intersection of perception and action in fields like robotics and autonomous systems. Its ability to unify a world model (video prediction) and a policy/planner within a single framework is a significant step toward more integrated AI agents.