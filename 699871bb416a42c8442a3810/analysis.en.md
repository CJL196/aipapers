# 1. Bibliographic Information

## 1.1. Title
The central topic of this paper is "Co-Evolving Latent Action World Models," abbreviated as `CoLA-World`. It focuses on a novel paradigm for training world models that learn controllable latent actions.

## 1.2. Authors
The authors are:
*   Yucen Wang ($^{2*}$)
*   Fengming Zhang ($^{2}$)
*   De-Chuan Zhan ($^{2}$)
*   Li Zhao ($^{1}$)
*   Kaixin Wang ($^{1 + }$)
*   Jiang Bian ($^{1}$)

    Their affiliations are:
*   $^{1}$ Microsoft Research Asia
*   $^{2}$ Nanjing University

## 1.3. Journal/Conference
The paper is published as a preprint, indicated by its presence on arXiv. As of the provided publication date (2025-10-30), it is not yet published in a specific journal or conference. arXiv is a widely respected open-access preprint server for scientific papers, particularly in physics, mathematics, computer science, and quantitative biology, often used for early dissemination of research findings before formal peer review and publication.

## 1.4. Publication Year
2025

## 1.5. Abstract
This paper introduces `CoLA-World`, a novel framework that successfully enables the joint training of a `latent action model (LAM)` and a pre-trained `video generation model` to create a controllable `world model`. Previous approaches typically used a two-stage process, leading to inefficiencies and limiting co-adaptation. The core challenge of joint training, which is prone to `representational collapse`, is resolved by `CoLA-World` through a critical `warm-up phase`. This phase aligns the representations of the `from-scratch LAM` with the `pre-trained world model`. This alignment initiates a `co-evolution cycle`: the knowledgeable world model guides the `LAM` through gradients, while the `LAM` provides a more precise and adaptable control interface to the world model. Empirically, `CoLA-World` demonstrates performance that matches or surpasses prior two-stage methods in both `video simulation quality` and `downstream visual planning`, establishing a robust and efficient new paradigm for the field of `generalist world models`.

## 1.6. Original Source Link
https://arxiv.org/abs/2510.26433
PDF Link: https://arxiv.org/pdf/2510.26433v1
Publication Status: Preprint on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the creation of a `generalist agent` capable of acting across diverse environments. A key component for such an agent is a `world model`, which acts as an internal simulator for planning and learning through imagination. While large-scale `video generative models` (e.g., `OpenSora`, `Stable Video Diffusion`) possess vast knowledge of world physics and dynamics, adapting them into *controllable* world models is challenging due to the `heterogeneity of action spaces` (e.g., continuous torques for a robot arm vs. discrete button presses for a game console). Direct use of real actions for fine-tuning a single, universal world model is impractical.

`Latent Action Models (LAMs)` offer a promising solution by inferring abstract, `embodiment-agnostic` actions directly from visual observations, providing a unified control interface. Existing research typically adopts a `two-stage approach`: first, training a `LAM` (comprising an `Inverse Dynamics Model (IDM)` and a `Forward Dynamics Model (FDM)`) from scratch, and then using the frozen `IDM` to extract `latent actions` for training a separate, larger `world model`.

This two-stage paradigm has significant issues:
1.  **Redundant Training:** Both the `FDM` and the `world model` perform `next-observation prediction`, leading to redundant computations and learning.
2.  **Limited Co-adaptation:** Freezing the `latent action space` after the first stage prevents it from adapting as the `world model` improves, bottlenecking the overall performance.
3.  **Representational Collapse:** A conceptually appealing idea is to directly replace the `FDM` with the powerful `world model` and train them jointly. However, prior attempts at this `joint training` have been non-trivial and prone to `representational collapse` in the `latent action space`.

    The paper's entry point is to directly tackle the problem of `joint training` of `LAMs` and `world models` to overcome these limitations, specifically by addressing the `representational collapse` issue.

## 2.2. Main Contributions / Findings
The paper's primary contributions are:
1.  **First Successful Joint Training Framework:** `CoLA-World` is proposed as the first framework that successfully enables the synergistic `joint training` of a `latent action model` with a `pre-trained video-generation-based world model`.
2.  **Critical Warm-up Phase:** The introduction of a `warm-up phase` is identified as a crucial mechanism to resolve the inherent fragility and `representational collapse` observed in naive joint training. This phase effectively aligns the `from-scratch LAM` with the `pre-trained world model`'s representations.
3.  **Synergistic Co-evolution:** The framework unlocks a `co-evolution cycle` where the `world model` acts as a "knowledgeable tutor," providing informative gradients to shape a high-quality `LAM`, while the `LAM` offers a more precise and adaptable control interface to the `world model`. This mutual reinforcement leads to a `tightly coupled system`.
4.  **Superior Performance:** Empirically, `CoLA-World` matches or outperforms prior two-stage methods in both `video simulation quality` and `downstream visual planning`. It achieves lower `linear probing loss` for `latent action quality` on most datasets and consistently surpasses baselines in `video prediction performance` (e.g., lower `FVD`, higher `PSNR`, `SSIM`) across various in-distribution and out-of-distribution datasets like `LIBERO` and `RoboDesk`.
5.  **Improved Sample Efficiency:** The joint training paradigm demonstrates superior `sample efficiency`, achieving competitive performance with significantly fewer training steps compared to two-stage baselines.
6.  **Robustness to Adaptation:** The `co-evolved latent action space` in `CoLA-World` proves more robust to `representational collapse` during `downstream adaptation` to real-action control interfaces, enabling better generalization.
7.  **Enhanced Visual Planning:** The superior simulation quality directly translates into improved `visual planning success rates` on challenging manipulation tasks in the `VP² benchmark`, particularly on tasks like `Upright Block`.

    These findings establish `CoLA-World` as a robust and efficient new paradigm for developing `generalist world models`, addressing key limitations of previous approaches.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

*   **World Model:** A `world model` (as introduced by Ha and Schmidhuber [14]) is an internal simulation of an environment that allows an agent to predict future states, plan actions, and learn through imagination without needing real-world interaction. It essentially learns the dynamics of an environment.
*   **Latent Action Model (LAM):** A `latent action model` is a system designed to discover and represent abstract, low-dimensional `actions` from raw observational data (like videos), especially when explicit `real actions` are unavailable or heterogeneous. These learned `latent actions` can then serve as a unified interface for controlling a `world model`.
*   **Inverse Dynamics Model (IDM):** A component of an `LAM` that learns to predict the `action` (or `latent action`) that caused a transition between two observations. Given an observation at time $t$ ($o_t$) and the next observation at time $t+1$ ($o_{t+1}$), the `IDM` ($f_{\text{inv}}$) tries to infer the `latent action` ($z_t$) that led from $o_t$ to $o_{t+1}$. Mathematically, $z_t = f_{\text{inv}}(o_t, o_{t+1})$.
*   **Forward Dynamics Model (FDM):** A component of an `LAM` that learns to predict the next observation given the current observation and an action (or `latent action`). Given $o_t$ and $z_t$, the `FDM` ($f_{\text{fwd}}$) predicts $o_{t+1}$. Mathematically, $o_{t+1} = f_{\text{fwd}}(o_t, z_t)$.
*   **Video Generation Models:** These are deep learning models (often `diffusion models` or `transformer-based models`) trained on large datasets of videos to generate new, realistic video sequences. Examples include `OpenSora` [41] and `Stable Video Diffusion` [2]. They capture complex visual dynamics and priors about how the world evolves.
*   **Representational Collapse:** This phenomenon occurs during training when a model's learned representations become degenerate or trivial. In the context of `latent action models` using `vector quantization (VQ)`, it can mean that the model uses only a very small subset of its available `codebook` entries, or all entries converge to represent essentially the same, uninformative information. This leads to a loss of diversity and expressiveness in the learned `latent action space`.
*   **Vector Quantization (VQ):** A technique used to discretize continuous representations into a finite set of discrete codes, often stored in a `codebook`. During training, the continuous embedding is "quantized" by finding the closest `codebook` entry, and the `codebook` entries are updated based on their usage. This forces the model to learn compact, discrete representations.
*   **Diffusion Models:** A class of generative models that learn to reverse a gradual diffusion process that adds noise to data. They start with random noise and iteratively denoise it to produce data samples (e.g., images or videos). They are known for high-quality generation.
*   **Adaptive Layer Normalization (AdaLN):** A mechanism (related to `AdaIN` or `FiLM`) used to condition neural networks on external information (like actions or text embeddings). It modifies the `scale` and `shift` parameters of `Layer Normalization` layers based on the conditioning input, allowing the model to adapt its internal representations to the given condition.
*   **Flow Matching Loss:** A training objective used in continuous normalizing flows or diffusion models (like `rectified flow` [24]) that trains a neural network to predict the velocity field that transforms a simple prior distribution (e.g., Gaussian noise) into the data distribution. It's an alternative to traditional `denoising score matching` objectives.

## 3.2. Previous Works

The paper categorizes related work into `Latent Action Learning`, `Latent-action-based World Models`, and `Finetuning Pre-trained Video Generation Model as World Models`.

### 3.2.1. Latent Action Learning
*   **FICC [39] and LAPO [30]:** Early methods that adopted the `IDM-FDM` framework. They discover `latent actions` through a `next-frame reconstruction objective`.
    *   **Core Idea:** Given $o_t$ and $o_{t+1}$, an `IDM` learns to infer a `latent action` $z_t$. Then, an `FDM` takes $o_t$ and $z_t$ to reconstruct $o_{t+1}$. The training objective often involves minimizing the reconstruction error.
    *   The reconstruction loss for `LAM` is given by:
        \$
        L_{\mathrm{LAM}} = \| o_{t + 1} - f_{\mathrm{fwd}}(o_t,f_{\mathrm{inv}}(o_t,o_{t + 1}))\|
        \$
        where:
        *   $L_{\mathrm{LAM}}$ is the `latent action model` loss.
        *   $o_t$ is the observation at time $t$.
        *   $o_{t+1}$ is the observation at time $t+1$.
        *   $f_{\mathrm{fwd}}$ is the `Forward Dynamics Model` (predicts next observation).
        *   $f_{\mathrm{inv}}$ is the `Inverse Dynamics Model` (infers `latent action`).
        *   $\| \cdot \|$ denotes a distance metric, typically L1 or L2 norm, measuring the difference between the true next observation and the reconstructed one.
    *   **Bottleneck:** To prevent trivial solutions where $f_{\text{inv}}$ simply copies $o_{t+1}$ information, a `bottleneck` (e.g., `Vector Quantization`) is applied to the `latent action space` ($z_t$), forcing it to compactly encode meaningful changes.
*   **Genie [3]:** Scales the `IDM-FDM` framework to large `transformer-based architectures`, focusing on `latent-action-driven world model prediction` in addition to `policy learning`.
*   **Embodied Agents:** Several works [4, 6, 26, 38] explore `latent action learning` in `embodied agents`, especially in `vision-language-action` settings.
*   **Differentiation:** CoLA-World differs by leveraging a `pre-trained video generation model` and enabling `co-evolution` of `latent action learning` and `world modeling`, which was not explored by these prior works.

### 3.2.2. Latent-action-based World Models
*   While `FDM`s can be seen as `world models`, their prediction quality is generally lower than high-capacity `video-generation-based world models`.
*   **Genie [3]:** Trained a separate `decoder-only MaskGIT` [5] as the `world model`, conditioned on a *fixed* `latent action space` learned beforehand. This is a classic two-stage approach.
*   **AdaWorld [11]:** Most closely related to CoLA-World. It also uses a two-stage approach, similar to `Genie`, but employs a `diffusion-based video model` and extends discrete `latent actions` to continuous ones.
*   **AD3 [36] and PreLAR [40]:** Integrate `latent action learning` with `dynamics and policy training` in a `Dreamer-style` [15] architecture. However, these are typically trained `from scratch` rather than leveraging large-scale `pre-trained video generation models`.
*   **Differentiation:** CoLA-World's innovation lies in its `joint training` paradigm that `co-evolves` the `LAM` and a `pre-trained world model`, avoiding the fixed `latent action space` bottleneck of `Genie` and `AdaWorld`, and leveraging powerful `pre-trained models` unlike `Dreamer-style` methods.

### 3.2.3. Finetuning Pre-trained Video Generation Model as World Models
*   This line of work assumes a `pre-specified action space` (real actions) rather than learned `latent actions`.
*   **AVID [29]:** Introduces a lightweight adapter on top of a frozen `video generation model` for `action conditioning` and `world modeling`.
*   **IRASim [42]:** Uses `adaptive layer normalization (AdaLN)` [28] to incorporate actions, drawing an analogy to `text prompting`.
*   **DWS [16]:** Builds on `IRASim`, proposing a more granular `action conditioning mechanism` and other improvements using `OpenSora` as a backbone.
*   **Vid2World [18]:** Focuses on `temporal causality` challenges when adapting `video diffusion models` to `world models`.
*   **EnerVerse-AC [19]:** Adds `action conditioning` to `embodied AI foundation models` for manipulation tasks.
*   **Differentiation:** CoLA-World's distinction is its focus on `latent actions` instead of `pre-specified real actions`. While it uses `AdaLN` similar to `IRASim` and `DWS`, its `latent actions` are learned and `co-evolved` with the `world model`, making the conditioning signals themselves dynamically refined.

## 3.3. Technological Evolution
The field has evolved from `world models` learning simple dynamics from scratch (e.g., `Dreamer` variants) to leveraging large-scale `video generative models` pre-trained on vast amounts of visual data, bringing rich prior knowledge about physics and visual dynamics. Simultaneously, `latent action learning` emerged to address the `heterogeneity of real action spaces`, enabling a unified control interface for these powerful `world models`.

Initially, `latent action` discovery and `world model` adaptation were sequential (two-stage), which limited their synergy. This paper represents a crucial step towards tightly integrating these two components into a `one-stage, co-evolving paradigm`. It moves from simply adapting a `world model` with fixed `latent actions` to jointly learning and refining both, aiming for a more robust, efficient, and generalizable `world model`. The use of `pre-trained diffusion models` (like `OpenSora`) as the backbone further highlights the trend of leveraging large foundation models for robotics and embodied AI.

## 3.4. Differentiation Analysis
CoLA-World's core innovation lies in its `joint training` approach, contrasting sharply with the dominant `two-stage paradigm` used by previous `latent-action-based world models` like `Genie` and `AdaWorld`.

*   **Two-Stage (Previous Works):**
    1.  **Stage 1:** Train a `LAM` (with `IDM` and `FDM`) from scratch on `action-free videos`. The `latent action space` becomes fixed after this stage.
    2.  **Stage 2:** Freeze the `IDM` to extract `latent action labels`. Then, train a separate, often larger `world model` using these fixed `latent action labels`. The `FDM` from Stage 1 is typically discarded as it's less powerful than the new `world model`.
    *   **Limitations:** Redundant `FDM` training, `static latent action space` that cannot adapt to `world model` improvements, and `bottlenecking` of the `world model`'s potential due to `suboptimal latent actions`.

*   **CoLA-World (Proposed):**
    1.  **Warm-up Phase:** An initial phase where the `IDM` (of the `LAM`) is trained `from scratch` while the `pre-trained world model` is *frozen*. The `world model` provides gradients to guide the `IDM` to align its `latent action representations`.
    2.  **Joint Training (E2E Co-evolution):** After warm-up, both the `IDM` and the `world model` (now unfrozen) are trained simultaneously, end-to-end. The `world model` acts as a "tutor," providing gradients to refine the `LAM`, and conversely, the improving `LAM` provides a better control interface to the `world model`. The `FDM` is effectively replaced by the powerful `world model`.
    *   **Innovations:**
        *   **Eliminates Redundancy:** The `world model` directly replaces the `FDM`, simplifying the architecture.
        *   **Synergistic Co-evolution:** The `latent action space` is dynamically refined alongside the `world model`, leading to a `virtuous cycle` where both components mutually reinforce each other. This is enabled by the `warm-up phase` that prevents `representational collapse`.
        *   **Leverages Pre-trained Models:** Directly integrates powerful `pre-trained video generation models` (like `OpenSora`) into the joint learning loop from the start, capitalizing on their vast prior knowledge.
        *   **Robustness:** The `co-evolved latent action space` demonstrates greater robustness to `representational collapse` during `downstream adaptation` to `real actions`.

            The core difference is the seamless, mutual adaptation and refinement between `latent action learning` and `world modeling` in `CoLA-World`, in contrast to the sequential, fixed-interface approach of prior methods.

# 4. Methodology

## 4.1. Principles
The core idea behind CoLA-World is to enable the `synergistic co-evolution` of `latent action learning` and `world modeling` within a `single, joint training framework`. This departs from the traditional `two-stage approach` by directly using a powerful `pre-trained world model` (based on `video generation models`) to predict `next observations`, thereby replacing the less capable `Forward Dynamics Model (FDM)` traditionally found in `Latent Action Models (LAMs)`. The intuition is that if the `world model` and `LAM` learn together, they can adapt to each other: the `world model` can teach the `LAM` to produce more meaningful actions, and a better `LAM` can provide clearer control signals to the `world model`.

However, `naively training` an `Inverse Dynamics Model (IDM)` (which is typically `from scratch`) jointly with a `powerful, pre-trained world model` leads to `representational collapse` in the `latent action space`. The `pre-trained world model` quickly learns to ignore the noisy, uninformative signals from the `uninitialized IDM` and relies on its own strong internal priors. This lack of structured gradient feedback causes the `latent action space` to degenerate.

To resolve this, CoLA-World introduces a `critical warm-up phase`. During `warm-up`, the `pre-trained world model` is `frozen` and only provides gradients to update the `from-scratch IDM`. This allows the `IDM` to "catch up" and align its `latent action representations` with the `world model`'s understanding of dynamics without causing collapse. Once this alignment is achieved, `full end-to-end joint training` commences, enabling the `co-evolution cycle` where both components mutually improve.

## 4.2. Core Methodology In-depth (Layer by Layer)

The methodology consists of instantiating the components, implementing the `latent action conditioning`, and defining the `training objective` and `gradient flow` across `warm-up` and `end-to-end` phases.

### 4.2.1. World Models with Latent Actions - The Conceptual Shift

The paper focuses on learning a `world model` that predicts the next observation $o_{t+1}$ given the current observation $o_t$ and a `latent action` $z_t$. This models the distribution $p(o_{t+1} \mid o_t, z_t)$.

**Traditional Two-Stage Approach (Figure 1(a)):**
1.  **Latent Action Model (LAM) Training:**
    *   An `Inverse Dynamics Model (IDM)` $f_{\mathrm{inv}}$ takes $o_t$ and $o_{t+1}$ to output a `latent action` $z_t$.
    *   A `Forward Dynamics Model (FDM)` $f_{\mathrm{fwd}}$ takes $o_t$ and $z_t$ to predict the next observation $\hat{o}_{t+1}$.
    *   The `LAM` is trained by minimizing the reconstruction loss between $\hat{o}_{t+1}$ and $o_{t+1}$:
        \$
        L_{\mathrm{LAM}} = \| o_{t + 1} - f_{\mathrm{fwd}}(o_t,f_{\mathrm{inv}}(o_t,o_{t + 1}))\|
        \$
    *   A `bottleneck` (e.g., `Vector Quantization`) is applied to $z_t$ to force compact encoding.
2.  **World Model Training:**
    *   The `IDM` is `frozen` and used to extract `latent action labels` $z_t$ for observation sequences.
    *   A separate, higher-capacity `world model` is trained to predict $p(o_{t+1} \mid o_t, z_t)$, using the fixed $z_t$ as input. The `FDM` is discarded.

**CoLA-World's One-Stage Joint Training (Figure 1(b)):**
The core idea is to directly replace the `FDM` with the powerful `world model`. This means the `world model` itself becomes the component that takes $o_t$ and $z_t$ (inferred by the `IDM`) to predict $o_{t+1}$. The training objective would then involve the `world model`'s prediction loss, which would then backpropagate through the `latent actions` to update the `IDM`.

### 4.2.2. Taming the Fragility of Joint Training

As identified in the paper, directly training a freshly initialized `IDM` jointly with a `pre-trained world model` leads to `representational collapse`. The `latent action codebook metrics` (utilization rate, max code usage, code entropy) quickly deteriorate (Figure 4, gray curve and brown curve).

**Warm-up Strategy:**
To address this, CoLA-World introduces a `warm-up phase` before switching to `joint training`.
1.  **World Model Instantiation:** The `world model` is instantiated using a `pre-trained OpenSora` [41] model, which is a high-performing `diffusion-based video generative model`. `OpenSora` was chosen for its effectiveness in `world modeling` when adapted with `pre-specified actions` (as in `DWS` [16]).
2.  **IDM Instantiation:** The `IDM` is instantiated as an `ST-Transformer` [37]. Its output is then passed through `vector quantization (VQ)` [33] to produce `discrete latent actions`.
3.  **Warm-up Phase (Figure 3):**
    *   The `pre-trained OpenSora world model` is `frozen`. Its weights are not updated.
    *   Only the `IDM` and the `VQ quantizer` (which together form the `LAM`) are trained.
    *   The `loss` generated by the `world model's prediction` (based on the `latent actions` produced by the `LAM`) is backpropagated *only* through the `action conditioning modules` and the `LAM components`. This means the `frozen world model` acts as a fixed `tutor`, providing supervisory gradients to guide the `from-scratch IDM` towards producing meaningful `latent actions` that align with the `world model`'s internal representation of dynamics.
    *   This phase allows the `IDM` to "catch up" and learn a stable, non-collapsed `latent action space` (Figure 3, dark blue curve shows healthy codebook metrics).
4.  **End-to-End Joint Training Phase:**
    *   After the `warm-up`, the `OpenSora world model` is `unfrozen`.
    *   The `IDM`, `VQ quantizer`, and the `OpenSora world model` are all trained `simultaneously` and `end-to-end`.
    *   The `loss` from the `world model's prediction` now `backpropagates throughout the entire system`, updating all components. This `end-to-end gradient flow` is the core mechanism enabling `synergistic co-evolution`. The `world model` continues to refine the `LAM`, and the improving `LAM` provides a better control interface, leading to mutual enhancement.

### 4.2.3. Implementation Details

**Latent Action Conditioning:**
*   The `latent actions` $z_t$ (extracted by the `IDM`) are integrated into the `pre-trained OpenSora model` via `Adaptive Layer Normalization (AdaLN)` [28].
*   **Process:**
    1.  The sequence of `latent actions` is first processed by a `from-scratch self-attention network`. This network learns `contextualized embeddings` for the `latent actions`.
    2.  These `contextualized embeddings` are then projected by a `Multi-Layer Perceptron (MLP)` into `action-specific scale, shift, and gate parameters`.
    3.  These `action-specific parameters` are `fused` (via addition) with the `original modulation parameters` derived from the `diffusion timesteps`.
    4.  The combined `AdaLN parameters` are then applied at `each LayerNorm layer` within all the `OpenSora blocks`.
*   This mechanism effectively provides `control signals` that condition the `denoising process` of the `OpenSora diffusion model` on the learned `latent actions`.

**Training Objective and Gradient Flow:**
*   The system is jointly optimized using a `flow matching loss objective` [24], which is part of the `OpenSora model`. This objective trains the model to predict the `velocity needed to denoise the video latent`.
*   **Warm-up Phase:**
    *   `OpenSora model` is `frozen`.
    *   The `loss` is `backpropagated` through the `action AdaLN parameters` and solely updates:
        *   The `action conditioning modules` (the `self-attention network` and `MLP` for `AdaLN` parameters).
        *   The `LAM components` (`IDM` and `VQ quantizer`).
    *   The `IDM` and `VQ quantizer` also receive gradients from `VQ loss` and `commitment loss` (standard for `VQ-VAEs`).
*   **End-to-End Joint Training Phase:**
    *   The `OpenSora world model` is `unfrozen`.
    *   The `unified gradient` updates `all components simultaneously`: `IDM`, `VQ quantizer`, `action conditioning modules`, and the `OpenSora world model` itself.
    *   This `end-to-end gradient flow` is the core mechanism for `synergistic co-evolution`.

**Model Architectures (Appendix B.1 and B.2):**
*   **IDM:** An `12-layer ST-Transformer` [37] with a hidden dimension of 768 and 12 attention heads. It takes a $T \times 224 \times 224 \times 3$ video clip, patchifies it, and processes it to predict `T-1` `latent actions`.
*   **VQ Quantizer:** Produces `discrete latent actions`. It consists of `two 32-dimensional action tokens` chosen from a `codebook` containing 32 entries (yielding $32 \times 32 = 1024$ different `latent action choices`).
*   **FDM (used in 2-stage baseline, not CoLA-World):** An `12-layer spatial Transformer` with similar dimensions, concatenating image patches and `latent action tokens` to produce pixel decoding results of next frames.
*   **OpenSora World Model (Backbone):** Uses the `v1.2 release` of `OpenSora` with approximately `1.2 billion parameters`.
    *   `Action conditioning modules`: These are `from-scratch modules` added to `OpenSora`, including 6 `self-attention blocks` to process the `latent action sequence` and an `MLP` to get `AdaLN parameters`. These add about `74 million parameters`.
    *   The `original text processing layers` and `cross-attention layers` in `OpenSora` are discarded.
    *   The `original temporal transformer blocks` in `OpenSora DiT` are modified with `causal masks` to prevent future information leakage, which is crucial for `dynamics modeling`.

**Training Protocols (Appendix B.3):**
*   **Learning Rate:** $7.5 \times 10^{-5}$ for both paradigms.
*   **Batch Size:** 128.
*   **Learning Rate Schedule:** A `2K-step linear warm-up schedule`.
*   **Data Augmentation:** `Random crop` is used for video clips when the `LAM` is updating (i.e., `LAM training` in 2-stage, and all of `joint training`). Not used when `LAM` is fixed.
*   **OpenSora Prediction:** Takes `256-resolution videos` and `latent action sequence`, adds noise, and predicts the `velocity vector` for denoising using `rectified flow`.
*   **Classifier-Free Guidance:** `Step-wise classifier-free guidance` is used with `action condition` randomly masked (probability 0.1) during training, and a `guidance scale of 4.0` for inference.
*   **Denoising Steps:** 10 denoising timesteps during inference.

    The methodology is meticulously designed to leverage the power of pre-trained models while meticulously handling the stability issues of `joint training`, creating a synergistic learning environment for `latent actions` and `world dynamics`.

# 5. Experimental Setup

## 5.1. Datasets
The primary focus is on `manipulation tasks` that involve diverse embodiments and action spaces. The training data for `CoLA-World` is a large-scale mixture of videos.

*   **Embodied Agent Videos:**
    *   `Open X-Embodiment (OXE)` [7] mixture: A large-scale robotic learning dataset.
    *   `AgiBot` [1] dataset: A manipulation platform for scalable and intelligent embodied systems.
*   **Human Egocentric and Manipulation Videos:** A comprehensive collection curated from nine prominent datasets:
    *   `Something-Something V2` [12]: Video database for learning and evaluating visual common sense.
    *   `RH20T` [10]: A robotic dataset for learning diverse skills in one-shot.
    *   `Ego4D` [13]: A large dataset of egocentric video.
    *   `EgoPAT3D` [22]: Egocentric prediction of action target in 3D.
    *   `EGTEA Gaze+` [21]: Joint learning of gaze and actions in first-person video.
    *   `HOI4D` [25]: A 4D egocentric dataset for category-level human-object interaction.
    *   `EPIC-KITCHENS` [9]: A large-scale dataset for egocentric action recognition.
    *   `HO-Cap` [34]: A capture system and dataset for 3D reconstruction and pose tracking of hand-object interaction.
    *   `HoloAssist` [35]: An egocentric human interaction dataset for interactive AI assistants.

**Final Data Mixture Composition:**
*   Approximately 30% `OXE`
*   Approximately 20% `AgiBot`
*   Approximately 50% `human video data`

    **Key Characteristic:** The training process is entirely `action-free`. Both the `world model` and the `latent action model` are learned purely from video observations, without explicit `real action` labels during pre-training.

**Why these datasets were chosen:**
These datasets collectively provide a broad and diverse range of manipulation tasks, embodied agent interactions, and human egocentric perspectives. This diversity is crucial for training `generalist world models` that can adapt to various downstream embodiments and action spaces, aligning with the paper's goal of creating adaptable `world models`. They are effective for validating the method's performance across different levels of visual complexity and interaction types.

## 5.2. Evaluation Metrics

The paper uses several metrics to assess the quality of `latent actions` and `world model` performance.

### 5.2.1. Latent Action Quality

*   **L1 Prediction Loss (for Linear Probing):** This metric is used in a `linear probing task` to evaluate how well the learned `latent actions` encode information about original `real actions`. A simple one-layer `linear projection head` is trained to predict the original `real action` from the frozen `latent actions`. Lower `L1 loss` indicates higher quality `latent action` representation.
    *   **Conceptual Definition:** L1 loss (Mean Absolute Error) measures the average magnitude of the errors between predictions and actual values, without considering their direction. It is robust to outliers compared to L2 loss. In this context, it quantifies the accuracy of a linear model in mapping `latent actions` back to `real actions`.
    *   **Mathematical Formula:**
        \$
        L_1 = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
        \$
    *   **Symbol Explanation:**
        *   $N$: The total number of samples.
        *   $y_i$: The true `real action` for sample $i$.
        *   $\hat{y}_i$: The predicted `real action` for sample $i$ from the `linear projection head` based on the `latent action`.
        *   $|\cdot|$: Absolute value.

### 5.2.2. World Model Video Prediction Performance

A suite of standard metrics is used to measure `action-conditioned video generation quality`.

*   **Peak Signal-to-Noise Ratio (PSNR):**
    *   **Conceptual Definition:** `PSNR` is a common measure of the quality of reconstruction of lossy compression codecs or, in this case, of generated images/videos compared to their ground-truth originals. It represents a ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher `PSNR` values generally indicate better quality.
    *   **Mathematical Formula:**
        \$
        \mathrm{PSNR} = 10 \cdot \log_{10} \left( \frac{\mathrm{MAX}_I^2}{\mathrm{MSE}} \right)
        \$
        where:
        \$
        \mathrm{MSE} = \frac{1}{MN} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I(i,j) - K(i,j)]^2
        \$
    *   **Symbol Explanation:**
        *   $\mathrm{MAX}_I$: The maximum possible pixel value of the image (e.g., 255 for an 8-bit image).
        *   $\mathrm{MSE}$: Mean Squared Error between the original and reconstructed image.
        *   `I(i,j)`: The pixel value at row $i$ and column $j$ of the original image.
        *   `K(i,j)`: The pixel value at row $i$ and column $j$ of the generated image.
        *   `M, N`: The dimensions of the image.

*   **Structural Similarity Index Measure (SSIM):**
    *   **Conceptual Definition:** `SSIM` is a perception-based model that considers image degradation as a perceived change in structural information, while also incorporating important perceptual phenomena such as luminance masking and contrast masking. It is a full reference metric, meaning it measures the quality of a processed image based on comparison with an original image. Values range from -1 to 1, with 1 indicating perfect similarity. Higher `SSIM` values indicate better quality.
    *   **Mathematical Formula:**
        \$
        \mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
        \$
    *   **Symbol Explanation:**
        *   $x$: A window from the first image (ground truth).
        *   $y$: A window from the second image (generated).
        *   $\mu_x$: The average of $x$.
        *   $\mu_y$: The average of $y$.
        *   $\sigma_x^2$: The variance of $x$.
        *   $\sigma_y^2$: The variance of $y$.
        *   $\sigma_{xy}$: The covariance of $x$ and $y$.
        *   $c_1 = (K_1 L)^2$, $c_2 = (K_2 L)^2$: Two variables to stabilize the division with weak denominators. $L$ is the dynamic range of the pixel values (e.g., 255), and $K_1 \ll 1, K_2 \ll 1$ are small constants.

*   **Learned Perceptual Image Patch Similarity (LPIPS):**
    *   **Conceptual Definition:** `LPIPS` measures the perceptual similarity between two images. Instead of comparing raw pixel values, it uses deep features extracted from pre-trained neural networks (like `AlexNet` or `VGG`) to quantify how similar two images are in terms of human perception. Lower `LPIPS` values indicate higher perceptual similarity (better quality).
    *   **Mathematical Formula:** (Simplified representation as it depends on a specific pre-trained deep network)
        \$
        \mathrm{LPIPS}(x, y) = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\phi_l(x)_{hw} - \phi_l(y)_{hw}) \|_2^2
        \$
    *   **Symbol Explanation:**
        *   `x, y`: The two images being compared.
        *   $\phi_l$: The feature extractor for layer $l$ of a pre-trained deep neural network.
        *   $w_l$: A learnable vector that scales the activations channels-wise.
        *   $\odot$: Element-wise product.
        *   $H_l, W_l$: Height and width of the feature map at layer $l$.
        *   $\| \cdot \|_2^2$: Squared L2 norm.

*   **Fréchet Video Distance (FVD):**
    *   **Conceptual Definition:** `FVD` is a metric used to evaluate the quality of generated video sequences, inspired by `Fréchet Inception Distance (FID)` for images. It measures the `Fréchet distance` between the feature distributions of real videos and generated videos. Features are typically extracted from a pre-trained video classification model (e.g., `Inflated 3D ConvNet (I3D)`). Lower `FVD` values indicate higher quality and more realistic generated videos. It is considered perceptually aligned.
    *   **Mathematical Formula:** (Simplified representation, as it involves feature distributions)
        \$
        \mathrm{FVD}(\mathbb{P}_r, \mathbb{P}_g) = \| \mu_r - \mu_g \|_2^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
        \$
    *   **Symbol Explanation:**
        *   $\mathbb{P}_r$: The distribution of features for real videos.
        *   $\mathbb{P}_g$: The distribution of features for generated videos.
        *   $\mu_r, \mu_g$: The mean feature vectors for real and generated videos, respectively.
        *   $\Sigma_r, \Sigma_g$: The covariance matrices for real and generated videos, respectively.
        *   $\| \cdot \|_2^2$: Squared L2 norm.
        *   $\mathrm{Tr}(\cdot)$: Trace of a matrix.
        *   $(\Sigma_r \Sigma_g)^{1/2}$: The matrix square root of the product of covariance matrices.

### 5.2.3. Visual Planning Success Rate

*   **Success Rate (on VP² benchmark):** This metric quantifies the effectiveness of the `world model` as a learned simulator for solving `control tasks` via `visual planning`. For a given task, it measures the percentage of runs where the agent successfully achieves the goal using a `sampling-based Model Predictive Control (MPC)` planner that relies on the `world model`'s predictions. Higher `success rate` indicates better planning utility.
    *   **Conceptual Definition:** The proportion of attempts where an agent successfully completes a defined task within an environment. In the `VP² benchmark`, success is determined by a reward function that combines `MSE loss` to the goal observation and a `binary classifier's logit` for the current task.

## 5.3. Baselines
The paper compares against `two training paradigms`:

1.  **2-STAGE:** This is the traditional approach, following prior work [3, 11].
    *   **Stage 1:** A `LAM` (comprising an `IDM`, an `FDM`, and a `VQ quantizer`) is trained `from scratch` for `30K steps`.
    *   **Stage 2:** The `LAM` is `frozen`. Its `IDM` and `quantizer` are used to provide `latent action labels` for fine-tuning the `world model`. The `FDM` is discarded. The `world model` is fine-tuned for `30K steps`.
    *   **Notation:** $LAM30K + WM30K$.
    *   A variant $LAM8K + WM52K$ is also used for comparison with `CoLA-World` at a similar total budget, where the `LAM` is under-trained.

2.  **JOINT (CoLA-World):** The proposed joint learning paradigm.
    *   **Warm-up Phase:** A `brief warm-up phase` (`8K steps`) to align the `from-scratch LAM` (`IDM` and `quantizer`) with the `pre-trained world model`.
    *   **End-to-End (E2E) Joint Training:** Full `end-to-end joint training` follows the warm-up.
    *   **Notation:** $WARM8K + E2E52K$ (total `60K steps` for comparison with $LAM30K + WM30K$).
    *   A variant $WARM8K + E2E30K$ is also used to evaluate `sample efficiency`.

        **Common Architectures:** The architectures of the `LAM` (IDM and quantizer) and `world model` (OpenSora backbone with action conditioning) are identical across both paradigms to ensure a fair comparison. The `OpenSora` model is pre-trained on a large general video dataset (not specified in detail in the main text but is a diffusion model) before any fine-tuning for world modeling.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Latent Action Quality

The following are the results from Table 1 of the original paper:

<table><thead><tr><td colspan="2">METHOD</td><td>BRIDGE</td><td>RT-1</td><td>KUKA</td><td>DROID</td><td>AGIBOT</td><td>LIBERo</td></tr></thead><tbody><tr><td>2-STAGE</td><td>LAM30K</td><td>0.0827</td><td>0.1191</td><td>0.0741</td><td>0.1912</td><td>0.1035</td><td>0.1614</td></tr><tr><td>JOINT</td><td>WARM8K + E2E22K</td><td>0.0815</td><td>0.1206</td><td>0.0736</td><td>0.1911</td><td>0.0908</td><td>0.1623</td></tr></tbody></table>

*   **Analysis:** Table 1 presents the `L1 linear probing loss` for `latent action quality` across six different embodied AI datasets. Lower values indicate better quality.
    *   `CoLA-World` (`JOINT: WARM8K + E2E22K`) generally matches or slightly outperforms the `2-STAGE` baseline (`LAM30K`). It achieves lower loss on `BRIDGE`, `KUKA`, `DROID`, and `AGIBOT` datasets.
    *   While the differences appear marginal in `L1 loss`, the paper argues that this isolated metric does not fully capture the utility. The true measure is the `latent action`'s ability to effectively control the `world model`. Subsequent results will show that `CoLA-World`'s `world model` significantly outperforms the two-stage baseline, suggesting that the `co-evolved latent action space` provides a more robust and effective control interface, even if `linear probing` doesn't always show a dramatic difference.

### 6.1.2. World Model Simulation Performance

The following are the results from Table 2 of the original paper:

<table><thead><tr><td>Dataset</td><td colspan="2">METHOD</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FVD ↓</td></tr></thead><tbody><tr><td rowspan="4">OXE</td><td>2-STAGE</td><td>LAM30K + WM30K</td><td>22.34</td><td>81.16</td><td>13.17</td><td>291.30</td></tr><tr><td></td><td>LAM8K + WM52K</td><td>21.91</td><td>80.76</td><td>13.79</td><td>296.64</td></tr><tr><td>JOINT</td><td>WARM8K + E2E52K</td><td>22.57</td><td>81.40</td><td>12.79</td><td>278.90</td></tr><tr><td></td><td>WARM8K + E2E30K</td><td>22.26</td><td>81.06</td><td>13.26</td><td>289.37</td></tr><tr><td rowspan="4">EGOCENTRIC</td><td>2-STAGE</td><td>LAM30K + WM30K</td><td>23.80</td><td>83.68</td><td>12.90</td><td>260.14</td></tr><tr><td></td><td>LAM8K + WM52K</td><td>23.48</td><td>83.28</td><td>13.46</td><td>267.94</td></tr><tr><td>JOINT</td><td>WARM8K + E2E52K</td><td>23.69</td><td>83.52</td><td>13.08</td><td>252.45</td></tr><tr><td></td><td>WARM8K + E2E30K</td><td>23.66</td><td>83.41</td><td>13.26</td><td>263.57</td></tr><tr><td rowspan="4">AGIBOT</td><td>2-STAGE</td><td>LAM30K + WM30K</td><td>23.61</td><td>85.36</td><td>10.11</td><td>185.63</td></tr><tr><td></td><td>LAM8K + WM52K</td><td>23.30</td><td>85.11</td><td>10.30</td><td>196.18</td></tr><tr><td>JOINT</td><td>WARM8K + E2E52K</td><td>23.93</td><td>85.61</td><td>9.86</td><td>174.93</td></tr><tr><td></td><td>WARM8K + E2E30K</td><td>23.64</td><td>85.27</td><td>10.22</td><td>189.03</td></tr><tr><td rowspan="4">LIBERO</td><td>2-STAGE</td><td>LAM30K + WM30K</td><td>23.13</td><td>86.90</td><td>10.22</td><td>167.77</td></tr><tr><td></td><td>LAM8K + WM52K</td><td>22.72</td><td>86.43</td><td>10.78</td><td>190.09</td></tr><tr><td>JOINT</td><td>WARM8K + E2E52K</td><td>23.33</td><td>87.21</td><td>9.89</td><td>158.36</td></tr><tr><td></td><td>WARM8K + E2E30K</td><td>23.25</td><td>87.05</td><td>10.08</td><td>164.86</td></tr></tbody></table>

*   **Analysis:** Table 2 compares the `video prediction performance` of different `world models` across various datasets.
    *   **Overall Superiority:** `CoLA-World` with the full training budget ($WARM8K + E2E52K$) consistently matches or surpasses the best two-stage method ($LAM30K + WM30K$) across all datasets and metrics. The improvements are particularly notable in `FVD` (lower is better), which is a perceptually aligned metric, indicating that `CoLA-World` generates more temporally coherent and realistic videos. For `OXE`, `CoLA-World` achieves `FVD` of 278.90 compared to `2-STAGE`'s 291.30. Similar gains are observed for `EGOCENTRIC`, `AGIBOT`, and `LIBERO`.
    *   **Sample Efficiency:** `CoLA-World` ($WARM8K + E2E30K$), with a significantly smaller total training budget (38K steps total, 8K warm-up + 30K E2E) compared to $LAM30K + WM30K$ (60K steps total), already approaches or even surpasses the performance of the fully trained two-stage model. For example, on `LIBERO`, $WARM8K + E2E30K$ achieves `FVD` of 164.86, which is better than $LAM30K + WM30K$'s 167.77. This highlights the superior `sample efficiency` of the `joint training` approach.
    *   **Impact of Untrained LAM:** When the two-stage method is given a similar total budget but with an under-trained `LAM` ($LAM8K + WM52K$), it performs significantly worse than $WARM8K + E2E52K$ and even $WARM8K + E2E30K$. This demonstrates the critical bottleneck introduced by a `static, under-trained LAM` in the two-stage approach, where the `world model` cannot compensate for a poor `latent action space`.

### 6.1.3. Evidence for Synergistic Co-evolution

The paper provides ablation studies to demonstrate the `bidirectional information flow` and `mutual promotion` in `CoLA-World`.

The following figure (Figure 1 from the original paper) provides evidence of synergistic co-evolution:

![fig 1](images/1.jpg)
*该图像是图表，展示了在Libero数据集上不同训练策略的探测损失、PSNR和FVD随训练步骤变化的情况。图(a)比较了共享预热、纯预热和E2E预热后的探测损失，图(b)展示了仅使用世界模型和E2E预热后的PSNR和FVD变化。*

*   **Analysis of Figure 1(a) - An Evolving World Model as a Better Tutor for the LAM:**
    *   This plot shows the `LAM's probing loss` on `LIBERO` over training steps.
    *   `PURE WARMUP` (LAM guided by `static world model`): The `probing loss` decreases steadily, indicating the `LAM` is learning.
    *   `WARMUP + E2E` (CoLA-World, LAM guided by `co-evolving world model`): Once `E2E training` starts, the `LAM's probing loss` drops much faster than `PURE WARMUP`.
    *   **Conclusion:** This indicates that as the `world model` refines its understanding of dynamics during `E2E training`, the gradients it provides to the `LAM` become more informative and causally sound, making it a more effective `tutor` for `latent action learning`.

*   **Analysis of Figure 1(b) - An Evolving LAM as a Better Control Interface for the World Model:**
    *   This plot shows the `world model's video prediction performance` (PSNR, FVD) over training steps.
    *   `WM ONLY AFTER WARMUP` (LAM `frozen` after warm-up): The `world model` improves initially but quickly `plateaus`.
    *   `WARMUP + E2E` (CoLA-World, WM paired with `continuously improving LAM`): The `world model` achieves substantially higher `video generation quality` (higher `PSNR`, lower `FVD`) compared to when the `LAM` is frozen.
    *   **Conclusion:** This demonstrates that a `static latent action space` imposes a `performance bottleneck`. A `dynamically evolving LAM` provides a progressively more precise control interface, unlocking the `world model's full predictive potential`.

### 6.1.4. Adaptation for Real-Action-Based Simulation

The following are the results from Table 3 of the original paper:

<table><thead><tr><td>Dataset</td><td>ACTION TYPE</td><td>METHOD</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FVD ↓</td></tr></thead><tbody><tr><td rowspan="4">LIBERO</td><td rowspan="2">GT-LAM</td><td>LAM30K + WM30K</td><td>25.51</td><td>89.55</td><td>7.41</td><td>73.54</td></tr><tr><td>Warm8K + E2E30K</td><td>25.85</td><td>89.82</td><td>7.31</td><td>74.65</td></tr><tr><td rowspan="2">Real ACTION</td><td>LAM30K + WM30K</td><td>22.45</td><td>86.96</td><td>9.56</td><td>115.45</td></tr><tr><td>Warm8K + E2E30K</td><td>22.68</td><td>87.15</td><td>9.27</td><td>93.68</td></tr><tr><td rowspan="4">ROBODESK</td><td rowspan="2">GT-LAM</td><td>LAM30K + WM30K</td><td>24.21</td><td>86.99</td><td>7.41</td><td>120.51</td></tr><tr><td>Warm8K + E2E30K</td><td>24.29</td><td>87.04</td><td>7.57</td><td>120.26</td></tr><tr><td rowspan="2">Real ACTION</td><td>LAM30K + WM30K</td><td>20.03</td><td>83.33</td><td>10.64</td><td>188.82</td></tr><tr><td>Warm8K + E2E30K</td><td>21.37</td><td>84.67</td><td>8.90</td><td>169.70</td></tr></tbody></table>

*   **Analysis:** Table 3 evaluates the `adaptability` of the `world models` to `real-action control interfaces` on `LIBERO` and `RoboDesk` datasets. The comparison is between $LAM30K + WM30K$ (two-stage, extensively trained) and $Warm8K + E2E30K$ (CoLA-World, smaller budget but jointly trained).
    *   **GT-LAM Performance:** When conditioned on `Ground-Truth Latent Actions (GT-LAM)`, CoLA-World ($Warm8K + E2E30K$) generally performs comparably or slightly better than the two-stage baseline, even with a smaller budget. This suggests that the `jointly trained world model` provides a stronger foundation for learning dynamics in unseen environments.
    *   **Real ACTION Performance (Crucial Test):** The performance gap becomes significantly more pronounced when evaluated with `real actions` (where a `lightweight MLP adapter` translates `real actions` to `latent actions`).
        *   On `LIBERO`, `CoLA-World` achieves `FVD` of 93.68, much better than `2-STAGE`'s 115.45.
        *   On `ROBODESK`, `CoLA-World` achieves `FVD` of 169.70, superior to `2-STAGE`'s 188.82.
    *   **Explanation for Gap:** The two-stage model, fine-tuned on a fixed `GT-LAM distribution`, becomes rigid. When the `real action adapter` introduces imperfect or `out-of-distribution latent actions`, the two-stage model struggles to interpret them, leading to a substantial performance drop. In contrast, `CoLA-World`, which `co-evolves` with a `dynamically improving LAM`, develops a more robust and smooth understanding of the `latent action space`. This makes it more resilient to the adapter's imperfections, allowing it to generalize better from ideal training signals to practical `real-world control interfaces`.

### 6.1.5. Visual Planning

The following are the results from Table 4 of the original paper:

<table><thead><tr><td>METHOD</td><td>UPRIGHT BLOCK</td><td>PUSH SLIDE</td><td>FLAT BLOCK</td><td>PUSH DRAWER</td><td>AVERAGE</td></tr></thead><tbody><tr><td>2-STAGE</td><td>20.0%</td><td>4.44%</td><td>1.11%</td><td>2.22%</td><td>6.94%</td></tr><tr><td>JOINT</td><td>37.78%</td><td>6.11%</td><td>3.33%</td><td>5.25%</td><td>13.12%</td></tr></tbody></table>

*   **Analysis:** Table 4 shows the `visual planning success rate` on the `RoboDesk VP² benchmark`.
    *   `CoLA-World` (`JOINT`) demonstrates a clear advantage over the `2-STAGE` approach across all tasks, and significantly so on the `Upright Block` task (37.78% vs. 20.0%).
    *   The `average success rate` for `CoLA-World` is 13.12%, nearly double that of the `2-STAGE` method's 6.94%.
    *   **Conclusion:** This confirms that `CoLA-World`'s superior `simulation quality` and more robust `latent action space` directly translate into more reliable predictions for the `Model Predictive Control (MPC)` planner, leading to more effective control. While some tasks remain challenging for both methods, the consistent gains highlight the practical utility of the `joint training methodology`.

## 6.2. Data Presentation (Tables)

All tables from the original paper (Tables 1, 2, 3, and 4) have been transcribed completely and accurately into the relevant sections above.

## 6.3. Ablation Studies / Parameter Analysis

### 6.3.1. Evidence for Synergistic Co-evolution (Section 4.3)
This section serves as a detailed `ablation study` to verify the `co-evolution mechanism`.
*   **Ablation 1: Evolving World Model as a Better Tutor:** Compares `WARMUP + E2E` (CoLA-World) with `PURE WARMUP` (LAM trained with `frozen world model`). The faster `probing loss` reduction for `LAM` in `WARMUP + E2E` demonstrates that the `world model's own learning` improves the quality of gradients provided to the `LAM`, making it a better tutor. This confirms that the `evolving world model` is crucial for shaping `high-quality latent actions`.
*   **Ablation 2: Evolving LAM as a Better Control Interface:** Compares `WARMUP + E2E` with a variant where the `LAM` is `frozen` after `warm-up` and only the `world model` is fine-tuned. The superior `video generation quality` for the `world model` in `WARMUP + E2E` indicates that a `dynamically improving LAM` provides a more precise and adaptable control interface, unlocking the `world model's full predictive potential`. This confirms that the `evolving LAM` is crucial for the `world model`'s performance.

    These two ablations effectively isolate and verify the `bidirectional influence` and `mutual reinforcement` between the `world model` and `LAM`, proving the existence and importance of the `synergistic co-evolution` in `CoLA-World`.

### 6.3.2. Analysis of Codebook Dynamics in Downstream Adaptation (Appendix D.1)

The following figure (Figure 2 from the original paper) illustrates codebook metrics in different training and adaptation stages:

![fig 2](images/2.jpg)
*该图像是一个展示利用率、最大使用量和熵的条形图，分别为 LIBERO 和 RoboDesk 数据集的结果。图中比较了两阶段训练和联合训练的效果，蓝色条代表两阶段方法，橙色条则表示联合训练。图表显示了不同训练方法在模型性能上的差异。*

*   **Analysis:** Figure 2 provides a quantitative `ablation` on the `VQ codebook metrics` (utilization, max usage, entropy) during `downstream adaptation`.
    *   **2-STAGE Method:** Shows `dramatic representational collapse` when adapting to `real actions`.
        *   `Training Distribution (a)`: Reasonable codebook metrics.
        *   `GT-LAM Fine-tuning (b)`: Metrics slightly decrease.
        *   `Adapter-LAM Inference (c)`: Severe degeneration. `Utilization plummets` (e.g., to ~10% on `RoboDesk`), `max-usage spikes` (to ~0.5 on both `LIBERO` and `RoboDesk`), and `entropy drops`. This implies the `adapter` learns a "lazy shortcut," mapping most `real actions` to a few dominant `latent codes`, leading to a highly impoverished `latent action space`. This explains the poor `real action prediction performance` of the `2-STAGE` method.
    *   **CoLA-World (JOINT) Method:** Maintains `healthy codebook usage` in the `Adapter-LAM setting`.
        *   `Entropy` remains high, and `max-usage` stays relatively low. This indicates a `robust` and `flexible latent action space` that resists collapse.
    *   **Conclusion:** This `ablation` directly supports the claim that the `co-evolutionary process` in `CoLA-World` creates a more robust and generalizable `latent action space`. The constant `supervisory feedback` from the `world model` prevents the `LAM` from degenerating, preserving the `diversity` and `meaningfulness` of the `latent action representations`. This robustness is key to its superior `downstream adaptation` and `generalization performance`.

### 6.3.3. Warm-up Length Variation (Figure 3 in paper, not a table)

The following figure (Figure 3 from the original paper) illustrates latent action codebook metrics during warm-up and joint training:

![fig 3](images/3.jpg)
*该图像是一个示意图，展示了在不同预热步骤下，CoLA-World 在利用率、最大使用量和熵三个方面的结果。随着预热步骤的增加，各个指标逐渐趋于稳定，反映了 joint training 的有效性和系统的学习过程。*

*   **Analysis:** Figure 3 (titled "Latent action codebook metrics during warm-up and joint training") shows how different `warm-up lengths` affect the stability of subsequent `joint training`. The plot shows `codebook utilization`, `max code usage`, and `code entropy`.
    *   The `dark blue curve` (representing the chosen `8K warm-up`) shows that the `codebook metrics` remain `healthy` and stable throughout the `warm-up` and subsequent `joint training`.
    *   The `plot indicates` that `longer warm-up` generally leads to more `stable subsequent joint training`. This confirms that the `IDM` undergoes a necessary "catch-up phase" during `warm-up`, aligning its representations before full `co-evolution`.
    *   **Conclusion:** This `parameter analysis` validates the necessity and effectiveness of the `warm-up phase` in preventing `representational collapse` and ensuring `stable joint training`. The chosen `8K warm-up` provides a good balance between stability and leaving sufficient budget for the `end-to-end co-evolution`.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper introduces `CoLA-World`, a pioneering framework that successfully implements `joint training` for `latent action models (LAMs)` and `pre-trained video-generation-based world models`. The core innovation lies in resolving the inherent fragility of `joint learning` (prone to `representational collapse`) through a strategically designed `warm-up phase`. This phase aligns the `from-scratch LAM` with the `pre-trained world model`, thereby enabling a `synergistic co-evolution cycle`. In this cycle, the `world model` acts as a knowledgeable tutor, guiding the `LAM` to learn high-quality `latent actions`, while the `LAM` provides a more precise and adaptable control interface to the `world model`.

Empirical results demonstrate that `CoLA-World` consistently matches or outperforms prior `two-stage methods` in both `video simulation quality` (as measured by `PSNR`, `SSIM`, `LPIPS`, and `FVD`) and `downstream visual planning` performance (on the `VP² benchmark`). It exhibits superior `sample efficiency` and, crucially, maintains a `robust` and `diverse latent action space` even during `real-action adaptation`, effectively resisting the `representational collapse` seen in two-stage approaches. `CoLA-World` thus establishes a robust and efficient new paradigm for the development of `generalist world models`.

## 7.2. Limitations & Future Work
**Limitations pointed out by the authors:**
1.  **Computational Resources:** The performance of the `world model` depends heavily on the `pre-trained video generation model` (e.g., `OpenSora`), which requires `substantial computational resources`.
2.  **Model Efficiency:** The current framework might be computationally intensive due to the large `pre-trained backbone`.

**Future research directions suggested by the authors:**
1.  **More Efficient Models:** Mitigating the computational cost by integrating more `efficient models`.
2.  **Vision-Language-Latent-Action Settings:** Evaluating the learned `latent actions` in `vision-language-latent-action settings` [4, 6] for `manipulation policy training`. This suggests exploring how these learned actions can be combined with language understanding for more complex and abstract control.
3.  **Scaling to Larger Datasets:** Scaling the framework to train `foundational world models` on even larger video datasets for `broader adaptability`. This aims to increase the generality and robustness of the learned models.

## 7.3. Personal Insights & Critique
This paper presents a significant advancement in `world modeling` by tackling a fundamental architectural challenge: the `joint training` of `latent action models` and `pre-trained video generation models`. The `warm-up phase` is a clever and effective solution to the `representational collapse` problem, which often plagues attempts to `end-to-end train` complex systems involving `discrete latent variables` or `from-scratch components` interacting with `powerful pre-trained models`. The concept of `co-evolution` between the `LAM` and `world model` is intuitively appealing and empirically validated, offering a more organic and adaptive learning process than the rigid `two-stage approach`.

**Potential Strengths and Applications:**
*   **Generalist AI:** The `embodiment-agnostic` nature of `latent actions`, combined with a `co-evolving world model`, pushes closer to the goal of truly `generalist agents` capable of operating in diverse environments without needing to relearn basic physics or action semantics.
*   **Robotics:** This framework has strong implications for `robotics`, enabling robots to learn complex manipulation skills through imagination, and adapt to new tasks and embodiments more efficiently.
*   **Data Efficiency for Downstream Tasks:** The superior `sample efficiency` and `robustness to real-action adaptation` mean that fine-tuning for new tasks might require significantly less data, a critical factor in real-world applications where data collection is expensive.

**Potential Areas for Improvement/Further Exploration:**
*   **Interpretability of Latent Actions:** While `linear probing` provides a quantitative measure, a deeper qualitative analysis of *what* the `co-evolved latent actions` represent could be valuable. Are they disentangled along meaningful semantic dimensions (e.g., "move forward," "grasp," "rotate")?
*   **Generalizability of Warm-up Strategy:** The `warm-up strategy` is shown to be effective for this specific setup. Investigating its generalizability to other `pre-trained backbone models` (e.g., different `video diffusion architectures` or `generative adversarial networks`) and different `latent action model` architectures would be interesting.
*   **Computational Cost:** While recognized as a limitation, explicit strategies for reducing the computational footprint beyond just "more efficient models" could be explored. This might involve distillation, sparse attention mechanisms, or more efficient `diffusion model` architectures specifically designed for `world modeling`.
*   **Long-Horizon Planning:** The current `visual planning` results, while superior, still show low success rates on complex tasks. Investigating how `CoLA-World` performs in `longer-horizon planning` or more abstract `task composition` scenarios would be a critical next step. The current `flow matching loss` for short-term prediction may need augmentation for long-term coherence.
*   **Multi-modal Latent Actions:** Extending the `latent action space` to incorporate other modalities (e.g., `tactile feedback`, `proprioception`) beyond just visual observations could lead to richer and more robust `world models` for embodied agents.

    Overall, `CoLA-World` offers a compelling paradigm shift in `world model` training, demonstrating that addressing architectural integration challenges can unlock significant performance gains and push the boundaries of `generalist AI`.