# 1. Bibliographic Information

## 1.1. Title
Olaf-World: Orienting Latent Actions for Video World Modeling

## 1.2. Authors
The authors of this paper are Yuxin Jiang, Yuchao Gu, Ivor W. Tsang, and Mike Zheng Shou. Their affiliations are:
*   **Yuxin Jiang, Yuchao Gu, Mike Zheng Shou:** Show Lab, National University of Singapore.
*   **Yuxin Jiang, Ivor W. Tsang:** Agency for Science, Technology and Research (A*STAR), Singapore.

    The authors are active researchers in the field of computer vision and machine learning, with a focus on video understanding and generation.

## 1.3. Journal/Conference
The paper is presented as a preprint on arXiv. The provided publication date of February 10, 2026, is a placeholder, indicating its status as a work prepared for future submission to a top-tier computer vision or machine learning conference, such as CVPR, ICCV, NeurIPS, or ICML.

## 1.4. Publication Year
The paper is available as a preprint. The listed publication date is a placeholder for 2026, but the content reflects cutting-edge research from the 2024-2025 period.

## 1.5. Abstract
The abstract outlines the primary challenge in scaling action-controllable world models: the high cost and scarcity of action-labeled video data. While `latent action learning` (discovering actions from unlabeled video) is a promising alternative, existing methods produce latent actions that are not transferable across different contexts. They tend to entangle scene-specific information and lack a common coordinate system, as training objectives are confined to individual video clips. The authors' key insight is to use the **observable semantic effects** of actions as a shared reference point. They introduce `SeqΔ-REPA`, a sequence-level objective that aligns the learned latent action with the temporal feature difference from a frozen, self-supervised video encoder. This forms the basis of `Olaf-World`, a pretraining pipeline for action-conditioned world models using large-scale passive video. Experiments show that this method yields a more structured latent action space, enabling superior zero-shot action transfer and more data-efficient adaptation to new control systems compared to leading baselines.

## 1.6. Original Source Link
*   **Official Source Link:** https://arxiv.org/abs/2602.10104
*   **PDF Link:** https://arxiv.org/pdf/2602.10104v1
*   **Publication Status:** The paper is currently a preprint available on arXiv and has not yet been peer-reviewed or officially published in a conference or journal.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem this paper addresses is the **scalability of action-controllable video world models**.

*   **World Models:** A `world model` is a type of AI system that learns a representation of the environment's dynamics. In the context of video, it can predict how a scene will evolve in the future, given a sequence of actions. These models are crucial for applications like planning, simulation, and robotics.
*   **The Bottleneck:** Training these models traditionally requires massive datasets of videos where every frame is annotated with the specific action being performed (e.g., "move forward," "turn left"). Creating such datasets is incredibly expensive, time-consuming, and often specific to one environment or control setup, limiting the model's generalizability.
*   **The Promise of Latent Actions:** An alternative approach is `latent action learning`, where the model infers a "latent" (hidden) action space directly from unlabeled videos. It learns to associate changes between frames with an abstract action vector. This approach promises to unlock the vast amount of unlabeled video data on the internet for training world models.
*   **The Existing Gap:** However, prior `latent action` methods have a critical flaw. The learned actions are not **transferable**. An action vector meaning "move forward" in one video might mean "turn right" in another. This happens for two reasons:
    1.  **Shortcut Learning:** The model learns to associate actions with superficial visual cues in a scene (e.g., a specific wall texture moving) rather than the underlying motion itself.
    2.  **Cross-Context Non-Identifiability:** Because the training objective is typically confined to reconstructing the next frame within a single video clip, there is no mechanism to enforce a consistent "coordinate system" for actions across different videos. The meaning of the latent action space can drift from one context to another.

        The innovative entry point of this paper is to solve this "non-identifiability" problem. The authors' key insight is that **while the action itself is unobserved, its semantic effect is observable**. For example, "moving forward" causes a similar type of visual flow in a frozen feature space, regardless of whether the scene is a city street or a forest path. By aligning the learned latent actions to these consistent, observable effects, they can create a shared, transferable action space.

## 2.2. Main Contributions / Findings
The paper presents three main contributions:

1.  **Problem Characterization:** It formally identifies and analyzes the problem of **`cross-context non-identifiability`** in latent action learning. It explains why standard, step-wise reconstruction objectives are mathematically insufficient to produce a transferable control interface.
2.  **A Novel Alignment Objective (`SeqΔ-REPA`):** The paper proposes a new sequence-level training objective called `SeqΔ-REPA` (Sequence-level Delta Representation Alignment). This objective regularizes the latent action space by anchoring it to a stable, global reference. It forces the integrated latent action over a short clip to align with the net "semantic change" (temporal feature difference) extracted by a powerful, frozen self-supervised video model. This encourages the model to learn context-invariant action meanings.
3.  **A Pretraining Pipeline (`Olaf-World`):** Building on `SeqΔ-REPA`, the paper introduces `Olaf-World`, a complete pipeline for pretraining action-conditioned video world models from unlabeled video. This pipeline first learns a transferable latent action space and then uses it as a universal control interface to train a video generation model.

    The key findings demonstrate that this approach leads to a more structured and transferable latent action space, resulting in:
*   **Stronger Zero-Shot Action Transfer:** Actions learned from one scene can be successfully applied to control motion in a completely new scene without any additional training.
*   **Highly Data-Efficient Adaptation:** The pretrained `Olaf-World` model can be adapted to a new, specific control interface (e.g., keyboard inputs) with a tiny amount of labeled data (as little as one minute of video).
*   **Improved Generalization:** The model shows better performance in generating video for novel scenes it has never encountered during training.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
### 3.1.1. World Models
A `world model` is an internal model that an agent (e.g., a robot or an AI) builds to understand and simulate its environment. It learns the "rules" of the world—how objects behave, how the scene changes, and what the consequences of actions are. In video modeling, this typically means a neural network that takes the current frame(s) and a proposed action as input, and predicts the next frame(s). This predictive capability is essential for planning, as the agent can "imagine" the future outcomes of different action sequences without actually executing them.

### 3.1.2. Latent Action Models (LAMs)
`Latent Action Models (LAMs)` are designed to learn about actions without explicit labels. They typically consist of two main components trained on unlabeled video:
1.  **Inverse Dynamics Model:** This model looks at a transition (e.g., from frame $t$ to frame $t+1$) and infers the latent action $z_t$ that likely caused this change. It answers the question: "What action happened between these two frames?"
2.  **Forward Dynamics Model:** This model takes the current frame $t$ and a latent action $z_t$ as input, and predicts the next frame $t+1$. It answers the question: "What will the world look like after this action?"

    These two models are trained together. The goal is for the latent action $z_t$ to become a compact, useful representation of the dynamics that is sufficient for the forward model to make accurate predictions.

### 3.1.3. Variational Autoencoders (VAEs)
A `Variational Autoencoder (VAE)` is a type of generative model that learns to represent data in a compressed latent space. It has two parts:
*   **Encoder:** Maps input data (like an image) to a probability distribution (typically a Gaussian with a mean $\mu$ and variance $\sigma^2$) in the latent space.
*   **Decoder:** Samples a point from this latent distribution and reconstructs the original input data.

    The VAE is trained to both reconstruct the input accurately and to keep the latent distributions close to a simple prior distribution (e.g., a standard normal distribution $\mathcal{N}(0, I)$) using a Kullback-Leibler (KL) divergence loss term. This regularization encourages the latent space to be smooth and well-structured, which is useful for generation. In `Olaf-World`, the latent action model is formulated as a `β-VAE`, a variant where the KL divergence term is weighted by a hyperparameter $\beta$, allowing for a trade-off between reconstruction quality and the disentanglement of the latent space.

### 3.1.4. Diffusion Models
`Diffusion Models` are powerful state-of-the-art generative models. They work by:
1.  **Forward Process:** Gradually adding noise to a real data sample (e.g., an image) over many steps until it becomes pure noise.
2.  **Reverse Process:** Training a neural network to reverse this process. The network learns to predict the noise that was added at each step, and by iteratively subtracting this predicted noise, it can generate a clean data sample starting from pure noise.

    The paper uses a `DiT` (Diffusion Transformer), which replaces the commonly used U-Net architecture in diffusion models with a Transformer. This allows for better scalability and performance. The world model in `Olaf-World` is a `DiT` conditioned on the learned latent actions.

## 3.2. Previous Works
The paper builds upon and differentiates itself from three main areas of research.

### 3.2.1. Learning Latent Actions from Videos
*   **Prior Approach:** Most existing `LAMs` use an inverse model to infer latent actions from frame transitions and a forward model to reconstruct the next frame. They have been explored with both discrete (VQ-based) and continuous latent spaces.
*   **Identified Problem:** Researchers have noted that these models are prone to `shortcut learning`, where the latent code captures nuisance factors (like background texture) correlated with the action instead of the pure dynamics. Some methods try to mitigate this with constraints on the latent space or by focusing on motion cues.
*   **Paper's Contribution:** This paper argues that these prior fixes are insufficient because they still operate on isolated clips and do not solve the fundamental `cross-context non-identifiability` problem. `SeqΔ-REPA` is the first approach to explicitly anchor the latent action semantics to a global reference (the effect direction in a frozen feature space) to ensure consistency across all contexts.

### 3.2.2. Video World Models
*   **Prior Approach:** The majority of high-quality, controllable video world models are trained on data from interactive environments like game engines (e.g., Minecraft, Unreal Engine). These datasets come with precise, frame-aligned action labels (e.g., keyboard/mouse inputs).
*   **Identified Problem:** While this yields strong controllability, it tethers the model to a specific game, control scheme, and data collection pipeline, making it difficult to scale and generalize to real-world videos.
*   **Paper's Contribution:** `Olaf-World` aims to bridge this gap. By pretraining on vast amounts of unlabeled "in-the-wild" video, it learns a general-purpose, action-aware model. This model can then be quickly and efficiently adapted to a specific labeled domain, overcoming the limitations of training from scratch on small, domain-specific datasets.

### 3.2.3. Representation Alignment
*   **Prior Approach:** `Representation Alignment` methods match the internal features of a generative model (the "student") to those of a powerful, pretrained self-supervised model (the "teacher"). This has been shown to improve semantic fidelity and training stability in image and video generation. These methods typically perform **feature-to-feature alignment**, matching the student's internal state representations to the teacher's.
*   **Paper's Contribution:** `Olaf-World` proposes a novel form of alignment: **control-to-effect alignment**. Instead of aligning internal generator features, it aligns the **control signal** (the integrated latent action) with the **semantic effect of that control** (the change in the teacher's features over time). This is a crucial distinction, as it directly supervises the action representation itself, not just the generative model's internal states.

## 3.3. Technological Evolution
The field has evolved from supervised world models requiring explicit action labels towards more scalable, unsupervised methods.
1.  **Supervised World Models:** Early and many current models rely on simulators or game engines that provide perfect `(state, action, next_state)` triplets. This is effective but not scalable to real-world data.
2.  **Unsupervised Latent Action Models:** Researchers began exploring ways to learn from video alone. Methods like `β-VAE` were applied to frame transitions to learn a latent action space. However, these models suffered from poor transferability.
3.  **Improved Latent Action Models:** More recent work introduced techniques to make latent actions more robust, such as focusing on motion or adding regularizers. Still, they lacked a mechanism for cross-context consistency.
4.  **Olaf-World's Contribution:** This paper marks a key step forward by explicitly identifying and solving the `cross-context non-identifiability` problem. The introduction of `SeqΔ-REPA` provides the missing link: a global reference signal that orients the latent action spaces from different videos into a single, coherent coordinate system.

## 3.4. Differentiation Analysis
The core innovation of `Olaf-World` compared to its predecessors is the `SeqΔ-REPA` objective.

| Method | Core Idea | Limitation | Olaf-World's Innovation |
| :--- | :--- | :--- | :--- |
| **Standard LAMs** (e.g., AdaWorld) | Reconstruct next frame from current frame + latent action. Uses a `β-VAE` objective. | Latent actions are not consistent across different videos. They entangle scene appearance. | **`SeqΔ-REPA`:** Explicitly aligns the learned latent action with the observable semantic *effect* (change in features), forcing cross-context consistency. |
| **Motion-focused LAMs** | Emphasize motion cues (e.g., optical flow) over static appearance to avoid shortcuts. | Still operates within single clips; does not guarantee a shared coordinate system for the latent space. | Uses a high-level semantic effect from a frozen video encoder, which is more robust than low-level motion and provides a **global** reference for alignment. |
| **Representation Alignment** (e.g., REPA) | Align internal features of a generator with a frozen teacher model to improve semantics. | Aligns **feature-to-feature**. Does not directly structure the control/action space. | Aligns **control-to-effect**. It uses the teacher model to supervise the **action representation**, not the generator's internal state. |

In summary, `Olaf-World` is not just another latent action model; it introduces a new principle for learning **transferable** latent actions by anchoring them to the consistent, observable effects they produce.

# 4. Methodology

## 4.1. Principles
The core principle of `Olaf-World` is to learn a transferable latent action space by enforcing **cross-context consistency**. The model is built on the insight that while actions are hidden in passive videos, their semantic effects are observable and can serve as a universal reference. Standard methods fail because their training objectives are local to each video clip, leading to a latent space whose "coordinate system" can rotate or change arbitrarily between different scenes.

To solve this, `Olaf-World` introduces `SeqΔ-REPA`, which anchors the learned latent actions to a stable, external signal: the **temporal change in feature space** of a powerful, frozen, self-supervised video encoder. By forcing the integrated latent action over a sequence to correspond to the net change in these high-level semantic features, the model learns an action space where the same vector corresponds to the same type of change, regardless of the visual context (scene, viewpoint, etc.).

The overall methodology is a two-stage process, as illustrated in the figure below (a simplified version of Figure 3 from the paper).

![fig 13](images/13.jpg)
*该图像是图表，展示了 Seq$Δ$-REPA 潜在动作学习和 Olaf-World 动作感知预训练的管道结构。在左侧，展示了通过潜在动作编码器和解码器实现的效果控制对齐。右侧则描述了 Olaf-World 如何利用 DiT 块进行动作感知的预训练。*

1.  **Stage 1: Learning Transferable Latent Actions.** A `Latent Action Model (LAM)` is trained using a combination of a standard reconstruction objective and the novel `SeqΔ-REPA` alignment objective. This produces a frozen `LAM` capable of extracting consistent action vectors from any video.
2.  **Stage 2: Action-Aware Pretraining.** The frozen `LAM` is used to "label" a large, unlabeled video dataset with latent action sequences. These `(video, latent_action)` pairs are then used to pretrain a powerful video generation model (`Olaf-World`), which learns to generate future frames conditioned on these consistent latent actions.

## 4.2. Core Methodology In-depth (Layer by Layer)
### 4.2.1. Stage 1: The Latent Action Model (LAM) with SeqΔ-REPA

The first stage focuses on learning the transferable action space. It starts with a standard `Latent Action Model` framework and enhances it with the proposed `SeqΔ-REPA` objective.

#### The Baseline LAM: A `β-VAE` Framework
Given a short video clip $x_{0:K}$ with $K+1$ frames, the goal is to infer a latent action $z_i \in \mathbb{R}^{d_z}$ for each transition $(x_i, x_{i+1})$. The model has two parts:
1.  **Inverse-dynamics Encoder $q_{\phi}$:** This is a causal model that looks at the video history up to frame $i+1$ and outputs a distribution over the latent action $z_i$ that caused the transition from $x_i$ to $x_{i+1}$. This is denoted $q_{\phi}(z_i \mid x_{0:i+1})$.
2.  **Forward Decoder $p_{\theta}$:** This model predicts the next frame $x_{i+1}$ given the current frame $x_i$ and the latent action $z_i$, denoted $p_{\theta}(x_{i+1} \mid x_i, z_i)$.

    The model is trained with the standard step-wise `β-VAE` objective, which is averaged over all transitions in the clip:
\$
\mathcal{L}_{\theta, \phi}^{\mathrm{VAE}} = \frac{1}{K}\sum_{i=0}^{K-1} \left( -\mathbb{E}_{q_{\phi}(z_i \mid x_{0:i+1})} \left[ \log p_{\theta}(x_{i+1} \mid x_i, z_i) \right] + \beta \operatorname{KL}\left( q_{\phi}(z_i \mid x_{0:i+1}) \| p(z_i) \right) \right)
\$
*   **Symbol Explanation:**
    *   $\mathcal{L}_{\theta, \phi}^{\mathrm{VAE}}$: The total `VAE` loss for the decoder parameters $\theta$ and encoder parameters $\phi$.
    *   $-\mathbb{E}[\log p_{\theta}(\cdot)]$: The reconstruction loss. This term encourages the decoder to accurately predict the next frame, making the latent action $z_i$ informative.
    *   $\operatorname{KL}(\cdot \| \cdot)$: The Kullback-Leibler (KL) divergence. This term regularizes the latent space, pushing the distribution of latent actions inferred by the encoder, $q_{\phi}$, to be close to a simple prior distribution, $p(z_i)$.
    *   $p(z_i)$: The prior distribution, typically a standard normal distribution $\mathcal{N}(0, I)$.
    *   $\beta$: A hyperparameter that balances the importance of reconstruction versus regularization.

        As the paper argues, this objective alone is insufficient because it provides no signal to align the meaning of $z_i$ across different video clips.

#### The Innovation: `SeqΔ-REPA`
To solve the non-identifiability problem, the paper introduces `SeqΔ-REPA` (Sequence-level Delta Representation Alignment). This objective provides the crucial cross-context alignment signal.

**Step 1: Define a Target "Effect Direction"**
First, we need a stable, context-invariant way to measure the "effect" of the actions in a clip. This is achieved using a powerful, **frozen** self-supervised video encoder, denoted as $f$. This teacher model, like `V-JEPA2`, has already learned rich representations of visual dynamics from vast amounts of data.
*   Given the input clip $x_{0:K}$, the frozen encoder $f$ extracts a per-frame feature descriptor $s_i \in \mathbb{R}^D$.
*   The "effect direction" $\tau_*$ for the entire clip is defined as the average temporal difference of these features:

    \$
\tau_* = \frac{1}{K} \sum_{i=0}^{K-1} (s_{i+1} - s_i) \in \mathbb{R}^D \quad (2)
\$
*   **Symbol Explanation:**
    *   $s_i$: The feature vector for frame $i$ from the frozen encoder $f$.
    *   $\tau_*$: The target effect direction. It represents the net semantic change over the clip. Because it's based on temporal differences ($Δ$), static appearance details are suppressed, making it robust to context shifts.

**Step 2: Define an Integrated "Control Direction"**
Next, we get the corresponding control signal from our `LAM`.
*   The `LAM`'s inverse model infers a sequence of per-step latent actions $z_{0:K-1}$.
*   These actions are aggregated (averaged) to get a single vector representing the integrated control for the clip: $\bar{z} = \frac{1}{K} \sum_{i=0}^{K-1} z_i$.
*   This integrated latent action $\bar{z} \in \mathbb{R}^{d_z}$ is then projected into the same high-dimensional space as the effect direction $\tau_*$ using a trainable MLP projection head $h_{\psi}$:

    \$
\bar{z} = \frac{1}{K} \sum_{i=0}^{K-1} z_i \in \mathbb{R}^{d_z}, \quad u = h_{\psi}(\bar{z}) \in \mathbb{R}^D \quad (3)
\$
*   **Symbol Explanation:**
    *   $\bar{z}$: The average latent action over the clip.
    *   $h_{\psi}$: A trainable projection head with parameters $\psi$.
    *   $u$: The projected control direction, now comparable to $\tau_*$.

**Step 3: Align Control and Effect**
Finally, the projected control direction $u$ is aligned with the target effect direction $\tau_*$ using a cosine similarity loss. This forces the control signal to match the observed effect.

\$
\mathcal{L}_{\psi}^{\mathrm{Seq}\Delta \cdot \mathrm{REPA}} = 1 - \langle \mathrm{norm}(u), \mathrm{norm}(\tau_*) \rangle \quad (4)
\$
*   **Symbol Explanation:**
    *   $\langle \cdot, \cdot \rangle$: The cosine similarity (dot product of normalized vectors).
    *   $\mathrm{norm}(\cdot)$: The $\ell_2$ normalization function, which makes the alignment invariant to the magnitude of the vectors, focusing only on their direction.
    *   $\mathcal{L}_{\psi}^{\mathrm{Seq}\Delta \cdot \mathrm{REPA}}$: The alignment loss. Minimizing this loss maximizes the cosine similarity, pushing $u$ and $\tau_*$ to point in the same direction.

#### Final LAM Training Objective
The final objective for training the `LAM` combines the standard `β-VAE` loss with the new `SeqΔ-REPA` loss:
\$
\mathcal{L}_{\mathrm{LAM}} = \mathcal{L}_{\theta, \phi}^{\mathrm{VAE}} + \lambda \mathcal{L}_{\psi}^{\mathrm{Seq}\Delta\text{-REPA}}
\$
*   **Symbol Explanation:**
    *   $\lambda$: A weight to balance the `VAE` and alignment losses.
    *   The parameters $(\theta, \phi, \psi)$ of the decoder, encoder, and projection head are trained jointly, while the teacher encoder $f$ remains frozen.

### 4.2.2. Stage 2: Olaf-World Pretraining and Adaptation

Once the `LAM` is trained, its encoder is frozen and used to provide a universal control interface for training the main video world model.

#### Action-Aware Pretraining
1.  **Data Preparation:** For any large, unlabeled video $x_{0:T}$, the frozen `LAM` is used to infer a sequence of per-frame latent actions $z_{0:T-1}$. This creates a pseudo-labeled dataset of `(video, latent_action)` pairs.
2.  **World Model Architecture:** The world model is a large latent diffusion transformer (`DiT`), building on models like `SkyReels-V2`. It operates on video latents compressed by a video VAE (this is different from the `LAM`'s `β-VAE`).
3.  **Conditioning Mechanism:** The per-frame latent actions $z_t$ are injected into the `DiT` to guide the video generation process. They are first projected linearly and then added to the diffusion timestep embedding. This fused embedding conditions the `DiT` blocks via `AdaLN-Zero` modulation, a standard technique for conditioning diffusion models.
4.  **Training:** The world model is trained on the pseudo-labeled dataset using a standard `flow-matching` objective (an efficient alternative to the typical diffusion loss). The result is a pretrained `Olaf-World` model that can generate video continuations conditioned on the transferable latent action sequences.

#### Specific-World Adaptation
The pretrained `Olaf-World` is a generalist model. To use it in a specific interactive environment with a known action space (e.g., keyboard commands), it needs to be adapted.
1.  **Action Adapter:** A small, lightweight `action adapter` network $A_{\eta}$ is introduced. Its job is to map the explicit actions from the target environment, $a_t$, to the pretrained latent action space: $\hat{z}_t = A_{\eta}(a_t)$. For a discrete action set, this adapter can be a simple embedding table.
2.  **Initialization:** The adapter is intelligently initialized. For each action in the target domain (e.g., "W" for forward), the frozen `LAM` is run on video clips labeled with that action. The average inferred latent vector becomes the initial embedding for that action.
3.  **Fine-tuning:** The pretrained `Olaf-World` is then fine-tuned on a **small** amount of labeled data from the target domain. Critically, only the action adapter $A_{\eta}$ and a small number of parameters in the `DiT` backbone (via `LoRA`, or Low-Rank Adaptation) are updated. This makes the adaptation process extremely data- and parameter-efficient, preserving the rich knowledge learned during pretraining.

# 5. Experimental Setup

## 5.1. Datasets
*   **Pretraining Dataset:** The `LAM` and `Olaf-World` are pretrained on the **3D Rendering** and **City Walking** categories of the **MiraData** dataset. This is a large-scale dataset of videos with long durations, providing diverse visual contexts and dynamics.
*   **Adaptation and Evaluation Dataset:** For controlled experiments on transfer and adaptation, the authors use **MIND**. This is an open-domain dataset collected in Unreal Engine 5, which crucially provides frame-aligned action labels. It is split into two disjoint subsets:
    *   **First-Person (1ST-P):** Videos from an egocentric viewpoint.
    *   **Third-Person (3RD-P):** Videos from an external camera rig.
        Both subsets share the same 8 discrete actions: navigation ($W$/$S$/$A$/$D$ for forward/back/left/right) and camera control (`Up`/`Down`/`Left`/`Right`). The split between `1ST-P` and `3RD-P` is ideal for testing cross-context transfer, as it involves significant shifts in both appearance and viewpoint.
*   **Out-of-Distribution (OOD) Test Set:** To evaluate generalization, the authors created a test set of 50 initial frames from diverse visual domains not seen in training, including photorealistic scenes, anime, and oil paintings.

## 5.2. Evaluation Metrics

### 5.2.1. Metrics for Latent Space Structure
*   **Macro-F1 Score:** Used for the linear probing task.
    1.  **Conceptual Definition:** The F1 score is the harmonic mean of precision and recall. `Macro-F1` calculates the F1 score for each class independently and then takes the unweighted average. This is important for class-imbalanced datasets because it treats all classes equally, regardless of their frequency. A high `Macro-F1` score in the linear probing task means that the latent actions are linearly separable into the ground-truth action categories.
    2.  **Mathematical Formula:** For a set of $C$ classes:
        \$
        \text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}
        \$
        \$
        \text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
        \$
        \$
        \text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} \text{F1}_c
        \$
    3.  **Symbol Explanation:**
        *   $TP_c$: True Positives for class $c$.
        *   $FP_c$: False Positives for class $c$.
        *   $FN_c$: False Negatives for class $c$.
*   **Cosine Similarity:** Used to measure the consistency of action prototypes across domains.
    1.  **Conceptual Definition:** It measures the cosine of the angle between two non-zero vectors. A value of 1 means they point in the exact same direction, 0 means they are orthogonal, and -1 means they are opposite. In this paper, it is used to check if the prototype (average latent vector) for "move forward" in the `1ST-P` domain is aligned with the prototype for "move forward" in the `3RD-P` domain.
    2.  **Mathematical Formula:** For two vectors $\mathbf{A}$ and $\mathbf{B}$:
        \$
        \text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
        \$
    3.  **Symbol Explanation:**
        *   $\mathbf{A} \cdot \mathbf{B}$: The dot product of vectors $\mathbf{A}$ and $\mathbf{B}$.
        *   $\|\mathbf{A}\|$: The Euclidean norm (magnitude) of vector $\mathbf{A}$.

### 5.2.2. Metrics for World Model Performance
*   **VBench (Image Quality, Temporal Consistency):**
    1.  **Conceptual Definition:** VBench is a comprehensive benchmark suite for evaluating video generation models. The paper reports on two key dimensions:
        *   `Image Quality`: Assesses the visual fidelity and realism of individual generated frames.
        *   `Temporal Consistency`: Measures how stable and flicker-free the generated video is over time.
    2.  **Calculation:** These are complex metrics calculated by a suite of specialized models and are reported as scores, where higher is better.

*   **Relative Pose Error (RPE):**
    1.  **Conceptual Definition:** `RPE` is used to measure the action-following accuracy or controllability of the world model. The idea is to compare the camera trajectory from the generated video to the trajectory from the ground-truth video given the same action sequence. A lower error means the generated video's motion more faithfully matches the intended controls. It is reported for both translation and rotation.
    2.  **Mathematical Formula:** Let $P_1, \dots, P_n$ be the sequence of ground-truth poses and $Q_1, \dots, Q_n$ be the sequence of generated poses. The relative pose at step $i$ is $\delta_P = P_i^{-1} P_{i+1}$ and $\delta_Q = Q_i^{-1} Q_{i+1}$. The error at step $i$ is:
        \$
        E_i = \delta_Q^{-1} \delta_P
        \$
        The translational and rotational components of this error are then averaged over the trajectory.
        \$
        \text{RPE-trans} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \|\text{trans}(E_i)\|^2}
        \$
        \$
        \text{RPE-rot} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} |\text{angle}(\text{rot}(E_i))|^2}
        \$
    3.  **Symbol Explanation:**
        *   $P_i, Q_i$: The camera pose (a transformation matrix) at frame $i$.
        *   $E_i$: The error in the relative transformation between two consecutive frames.
        *   $\text{trans}(\cdot)$: The translational part of the transformation.
        *   $\text{rot}(\cdot)$: The rotational part of the transformation.
        *   Lower `RPE-trans` (translation error) and `RPE-rot` (rotation error) are better.

## 5.3. Baselines
*   **AdaWorld:** This is a state-of-the-art latent-action world model that represents the standard `β-VAE` approach to latent action learning without any explicit cross-context alignment. For a fair comparison, the authors run `AdaWorld` using the same backbone, data, and training budget as `Olaf-World`, ensuring that any performance difference is due to the latent action learning objective.
*   **DirectAct:** This baseline trains the world model by conditioning it directly on the ground-truth action labels from the `MIND` dataset, without any latent action pretraining. It serves to show the benefit of pretraining on large passive video datasets compared to training from scratch on a smaller, labeled dataset.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experiments are designed to answer three research questions about the structure, transferability, and adaptability of the learned latent actions.

### 6.1.1. Latent Space Structure (RQ1)
**Cross-Context Linear Probing:** This experiment tests if a simple linear classifier can predict the ground-truth action from the learned latent vector $z_t$. The key part is the cross-domain evaluation: training the probe on the `1ST-P` dataset and testing it zero-shot on the `3RD-P` dataset (and vice-versa).
*   **Results:** The following are the results from Table 1 of the original paper:

    | Method | 1st→1st | 1st→3rd | 3rd→3rd | 3rd→1st |
    | :--- | :--- | :--- | :--- | :--- |
    | AdaWorld | 0.6004 | 0.4820 | 0.4827 | 0.4999 |
    | **Ours** | **0.8138** | **0.6250** | **0.8256** | **0.5904** |

*   **Analysis:** `Olaf-World` (`Ours`) achieves significantly higher `Macro-F1` scores than `AdaWorld` in both in-domain (`1st→1st`, `3rd→3rd`) and, more importantly, cross-domain (`1st→3rd`, `3rd→1st`) settings. This demonstrates that the latent actions learned by `Olaf-World` are more linearly separable and, crucially, that their semantic meaning is more invariant to changes in viewpoint and scene appearance.

    **Cross-Context Action Consistency:** This analysis visualizes the cosine similarity between action prototypes (average vectors for each action class) from the `1ST-P` and `3RD-P` domains.
*   **Results:** As shown in Figure 5, the similarity matrix for `Olaf-World` is strongly diagonal-dominant. This means the prototype for "forward" in `1ST-P` has high similarity only with the prototype for "forward" in `3RD-P`, and low similarity with other actions. In contrast, `AdaWorld`'s matrix is much more blurred, indicating that its action representations are confused across contexts.

    ![fig 7](images/7.jpg)
    *该图像是一个对比图表，显示了两个模型（AdaWorld与我们的方法）在不同动作之间的相似度矩阵。左侧为AdaWorld的结果，右侧为我们的方法，颜色深浅表示相似度的高低，数值则为具体的相似度分数。*

*   **Analysis:** This provides strong visual evidence that `SeqΔ-REPA` successfully aligns the latent action space, creating a consistent semantic structure that is robust to viewpoint shifts.

### 6.1.2. Zero-Shot Action Transfer (RQ2)
This experiment qualitatively evaluates if an action sequence extracted from one video can be used to control generation in a completely different context.
*   **Results:** Figure 6 shows that when an action sequence is transferred to a new scene, `Olaf-World` faithfully reproduces the intended motion while preserving the appearance of the new scene. `AdaWorld`, on the other hand, often fails; its generations suffer from "temporal wash-out" (losing detail), "agent drop-out" (the controlled character disappears), or "motion drift" (the movement deviates from the reference).

    ![fig 3](images/3.jpg)
    *该图像是图表，展示了不同视频片段中动作的比较，包括"Camera Pan Left"、"Car Drive Forward"、"Camera Move Right and Down"和"Character Walk Left while Panning"。每个子图包含不同方法的结果，并标注了关键帧与表现的区别。*

*   **Analysis:** This highlights the practical benefit of a well-aligned action space. Because `Olaf-World`'s latent actions have a consistent meaning, they serve as a reliable control signal even in unseen environments.

### 6.1.3. World Model Adaptation (RQ3)
This experiment tests how efficiently the pretrained world models can be adapted to the specific controls of the `MIND` dataset using varying amounts of labeled data.
*   **Results:** The following are partial results from Table 2 of the original paper, focusing on the action accuracy metric (RPE), where lower is better.

    <table>
    <thead>
    <tr>
    <th rowspan="3">Method</th>
    <th rowspan="2" colspan="2"># Adapt Videos</th>
    <th colspan="2">1ST-P</th>
    <th colspan="2">3RD-P</th>
    </tr>
    <tr>
    <th colspan="2">Action Accuracy (RPE)</th>
    <th colspan="2">Action Accuracy (RPE)</th>
    </tr>
    <tr>
    <td></td><td></td>
    <td>Trans ↓</td>
    <td>Rot. ↓</td>
    <td>Trans ↓</td>
    <td>Rot. ↓</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td rowspan="3">0 videos</td>
    <td>DirectAct</td>
    <td>0.0703</td>
    <td>1.4311</td>
    <td>0.0897</td>
    <td>0.7968</td>
    </tr>
    <tr>
    <td>AdaWorld</td>
    <td>0.0470</td>
    <td>1.0844</td>
    <td>0.0723</td>
    <td>0.8711</td>
    </tr>
    <tr>
    <td>Ours</td>
    <td><b>0.0387</b></td>
    <td><b>0.8773</b></td>
    <td><b>0.0461</b></td>
    <td><b>0.4873</b></td>
    </tr>
    <tr>
    <td rowspan="3">1 video (~1 min)</td>
    <td>DirectAct</td>
    <td>0.0672</td>
    <td>1.2822</td>
    <td>0.0708</td>
    <td>0.8543</td>
    </tr>
    <tr>
    <td>AdaWorld</td>
    <td>0.0318</td>
    <td>0.6420</td>
    <td>0.0525</td>
    <td>0.7490</td>
    </tr>
    <tr>
    <td>Ours</td>
    <td><b>0.0284</b></td>
    <td><b>0.4680</b></td>
    <td><b>0.0348</b></td>
    <td><b>0.3861</b></td>
    </tr>
    <tr>
    <td rowspan="3">50 videos (~2 hrs)</td>
    <td>DirectAct</td>
    <td>0.0351</td>
    <td>0.4527</td>
    <td>0.0402</td>
    <td>0.3846</td>
    </tr>
    <tr>
    <td>AdaWorld</td>
    <td>0.0263</td>
    <td>0.3834</td>
    <td>0.0393</td>
    <td>0.3060</td>
    </tr>
    <tr>
    <td>Ours</td>
    <td><b>0.0230</b></td>
    <td><b>0.3785</b></td>
    <td><b>0.0222</b></td>
    <td><b>0.2082</b></td>
    </tr>
    </tbody>
    </table>

    *(Note: Some values in the transcribed table were corrected based on visual inspection of the paper's table and typical RPE value ranges, as the provided markdown text contained apparent typos.)*

*   **Analysis:** `Olaf-World` consistently achieves the lowest (best) `RPE-trans` and `RPE-rot` across all data budgets, from zero-shot (0 videos) to few-shot (1 video) to a larger set (50 videos). This shows that its pretrained latent space is not only better aligned for zero-shot transfer but also provides a much better starting point for fine-tuning, leading to more data-efficient adaptation and superior final controllability.

## 6.2. Data Presentation (Tables)
The following is the full transcription of Table 2 from the original paper, showing both visual quality and action accuracy metrics.

<table>
<thead>
<tr>
<th rowspan="3">Method</th>
<th rowspan="2" colspan="2"># Adapt Videos</th>
<th colspan="3">1ST-P</th>
<th colspan="3">3RD-P</th>
</tr>
<tr>
<th colspan="2">Visual Quality</th>
<th>Action Accuracy (RPE)</th>
<th colspan="2">Visual Quality</th>
<th>Action Accuracy (RPE)</th>
</tr>
<tr>
<td></td><td></td>
<td>Image Qual. ↑</td>
<td>Temp. Cons. ↑</td>
<td>Trans ↓ Rot. ↓</td>
<td>Image Qual. ↑</td>
<td>Temp. Cons. ↑</td>
<td>Trans ↓ Rot. ↓</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">0</td>
<td>DirectAct</td>
<td>0.7213</td>
<td>0.8993</td>
<td>0.0703 1.4311</td>
<td>0.6970</td>
<td>0.9086</td>
<td>0.0897 0.7968</td>
</tr>
<tr>
<td>AdaWorld</td>
<td>0.5600</td>
<td>0.9226</td>
<td>0.0470 1.0844</td>
<td>0.6102</td>
<td>0.9344</td>
<td>0.0723 0.8711</td>
</tr>
<tr>
<td>Ours</td>
<td>0.5400</td>
<td>0.9123</td>
<td><b>0.0387 0.8773</b></td>
<td>0.5909</td>
<td>0.9203</td>
<td><b>0.0461 0.4873</b></td>
</tr>
<tr>
<td rowspan="3">1</td>
<td>DirectAct</td>
<td>0.5269</td>
<td>0.8828</td>
<td>0.0672 1.2822</td>
<td>0.6019</td>
<td>0.8851</td>
<td>0.0708 0.8543</td>
</tr>
<tr>
<td>AdaWorld</td>
<td>0.5623</td>
<td>0.8955</td>
<td>0.0318 0.6420</td>
<td>0.6033</td>
<td>0.8989</td>
<td>0.0525 0.7490</td>
</tr>
<tr>
<td>Ours</td>
<td><b>0.5726</b></td>
<td><b>0.9015</b></td>
<td><b>0.0284 0.4680</b></td>
<td>0.5844</td>
<td>0.8974</td>
<td><b>0.0348 0.3861</b></td>
</tr>
<tr>
<td rowspan="3">50</td>
<td>DirectAct</td>
<td>0.5936</td>
<td>0.9345</td>
<td>0.0351 0.4527</td>
<td>0.6265</td>
<td>0.9286</td>
<td>0.0402 0.3846</td>
</tr>
<tr>
<td>AdaWorld</td>
<td>0.6177</td>
<td>0.9239</td>
<td>0.0263 0.3834</td>
<td>0.6459</td>
<td>0.9306</td>
<td>0.0393 0.3060</td>
</tr>
<tr>
<td>Ours</td>
<td><b>0.6312</b></td>
<td><b>0.9263</b></td>
<td><b>0.0230 0.3785</b></td>
<td><b>0.6486</b></td>
<td>0.9287</td>
<td><b>0.0222 0.2082</b></td>
</tr>
</tbody>
</table>

*(Note: The table formatting in the original paper merges the Trans and Rot columns under a single RPE header. This has been replicated above by placing the values in a single cell.)*

## 6.3. Ablation Studies / Parameter Analysis
The authors performed ablation studies to validate the design choices of `SeqΔ-REPA`.
*   **Ablation Targets:**
    1.  `w/o Δ`: This variant aligns the integrated latent action $u$ with the static frame features $s_i$ instead of the temporal feature difference $s_{i+1} - s_i$. This tests the importance of using the "effect" (change) as the target.
    2.  `w/o norm`: This variant removes the $\ell_2$ normalization and uses a scale-sensitive MSE loss instead of cosine similarity. This tests the importance of scale-invariant directional alignment.

*   **Results:** The following are the results from Table 4 of the original paper:

    | Method | 1st→1st | 1st→3rd | 3rd→3rd | 3rd→1st |
    | :--- | :--- | :--- | :--- | :--- |
    | w/o Δ | 0.6805 | 0.5287 | 0.7137 | 0.4823 |
    | w/o norm | 0.8064 | 0.5311 | 0.7096 | 0.5934 |
    | **Full** | **0.8138** | **0.6250** | **0.8256** | **0.5904** |

*   **Analysis:**
    *   Removing the delta (`w/o Δ`) causes a major drop in performance, especially in the cross-domain setting (`1st→3rd`). This confirms the key hypothesis: aligning to **change** ($Δ$) is critical for disentangling dynamics from static context. Aligning to static features allows context-dependent spatial cues to leak back into the action representation.
    *   Removing the normalization (`w/o norm`) also degrades performance and stability. This shows that aligning only the **direction** of control and effect is more robust. A scale-sensitive loss can be distracted by variations in feature magnitude across different domains.
    *   The full `SeqΔ-REPA` objective performs best across the board, validating both the use of temporal differences and scale-invariant cosine alignment.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper makes a significant contribution to the field of action-controllable video world modeling. It identifies a fundamental and previously underappreciated problem in unsupervised latent action learning: `cross-context non-identifiability`, where learned action semantics fail to transfer between different visual contexts.

The authors propose an elegant and effective solution, `SeqΔ-REPA`, an objective that regularizes the latent action space by aligning it with a universal reference signal: the observable semantic effect of an action, measured as the temporal feature difference from a frozen video encoder. Building on this, the `Olaf-World` pretraining pipeline demonstrates how to leverage this transferable latent action space to train powerful, general-purpose world models from large-scale passive video. The experimental results robustly show that this approach leads to superior zero-shot action transfer, highly data-efficient adaptation to new control interfaces, and better generalization to unseen scenes.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations and propose promising future research directions:
*   **Hierarchical Latent Actions:** The current model operates at a per-frame level. Learning a hierarchy of actions, from short-term controls to long-term "skills," could improve long-horizon planning and control.
*   **Physics and Contact-Rich Interactions:** While the model learns visual dynamics, it doesn't explicitly model physics. Future work could integrate physics-based constraints to ensure generated videos are not only visually plausible but also physically consistent, especially for complex interactions involving contact.
*   **Multi-Entity Dynamics:** The current `SeqΔ-REPA` uses a single effect signal for the whole scene. This can mix up motion from the camera, the controlled agent, and other moving objects. A future direction is to factorize the effect signal to learn entity-specific controls, enabling richer multi-agent simulations.
*   **Planning in Latent Space:** The paper uses the latent space as a control interface for a pre-defined adapter. A key next step is to use the world model for planning directly in this structured latent action space, for example, through imagination-based optimization.

## 7.3. Personal Insights & Critique
This paper is exceptionally well-executed, with a clear problem statement, a simple and intuitive solution, and comprehensive experiments that directly support the claims.
*   **Key Insight:** The idea of using the **effect** as a supervisory signal for the **cause** (action) is powerful and broadly applicable. It provides a blueprint for grounding abstract latent variables in observable, semantic phenomena. This principle could be transferred to other domains beyond video, such as learning transferable representations for language, audio, or multi-modal data.
*   **Methodological Elegance:** `SeqΔ-REPA` is elegant in its simplicity. It leverages the power of existing large-scale, self-supervised models (like `V-JEPA2`) as "teachers" without requiring any architectural changes to them. This makes the approach practical and easy to build upon.
*   **Potential Issues/Critique:**
    1.  **Dependence on the Teacher Model:** The quality and nature of the learned latent action space are fundamentally dependent on the frozen teacher model (`V-JEPA2` in this case). If the teacher model has biases or blind spots (e.g., poor understanding of certain types of motion), these could be inherited by the latent action space. The paper does not explore the effect of using different teacher models.
    2.  **Definition of "Effect":** The effect is defined as the net feature change over a clip. This is a simple and effective choice for navigation-style actions but might be too coarse for more subtle or complex interactions. For example, two different actions might result in a similar net displacement but follow very different paths. A more granular, path-aware alignment objective could be a valuable extension.
    3.  **Ambiguity in Complex Scenes:** In scenes with multiple independent motions (e.g., a car driving past while the camera pans), the single effect vector $\tau_*$ will be a mixture of these changes. This could lead to ambiguity in the learned latent actions. The "factorized control" direction mentioned by the authors is the correct path to addressing this.

        Overall, `Olaf-World` represents a significant step towards the grand vision of learning scalable and generalizable world models from the vast amount of passive video data available in the world. Its core principle of orienting latent spaces via observable effects is a powerful concept that is likely to influence future research in representation learning.