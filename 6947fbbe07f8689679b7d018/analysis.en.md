# 1. Bibliographic Information

## 1.1. Title
V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

## 1.2. Authors
Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Geji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas.
Affiliations: FAIR (Fundamental AI Research) at Meta, Mila – Quebec AI Institute, and Polytechnique Montréal.

## 1.3. Journal/Conference
The paper is published as a preprint on arXiv (arXiv:2506.09985) and originates from the FAIR team at Meta. Given the authors (including Yann LeCun) and the scale of the work, it is likely targeted for a top-tier computer vision or machine learning conference such as CVPR or NeurIPS.

## 1.4. Publication Year
Published on June 11, 2025 (UTC).

## 1.5. Abstract
The paper explores a self-supervised approach called V-JEPA 2 that combines internet-scale video data with a small amount of robot interaction data to develop world models. These models are capable of understanding, predicting, and planning in physical environments. Pre-trained on over 1 million hours of video, V-JEPA 2 achieves state-of-the-art (SOTA) performance on motion understanding, action anticipation, and video question-answering. Furthermore, a version called V-JEPA 2-AC, trained on a small robot dataset, enables zero-shot picking and placing on physical robots without environment-specific training or rewards.

## 1.6. Original Source Link
[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)

---

# 2. Executive Summary

## 2.1. Background & Motivation
A fundamental goal in AI is to create agents that learn about the world much like humans do—largely through observation rather than just trial and error. Traditional methods for training robots often rely on massive amounts of specific interaction data or explicit reward signals (reinforcement learning), which are expensive and difficult to scale. 

The core problem is the lack of a "world model"—an internal representation that allows an agent to predict the future consequences of its actions. While previous work used video generation (predicting every pixel), this is computationally expensive and focuses on irrelevant details (like the movement of every leaf on a tree). V-JEPA 2 aims to solve this by predicting in a **latent space** (a compact mathematical representation), focusing on high-level dynamics rather than pixel-perfect reconstruction.

## 2.2. Main Contributions / Findings
1.  **V-JEPA 2 Model:** A large-scale video model (up to 1 billion parameters) trained on 1.1 million hours of video using a self-supervised objective.
2.  **State-of-the-Art Understanding:** The model achieves SOTA results on benchmarks requiring fine-grained motion understanding (Something-Something v2) and temporal reasoning (PerceptionTest).
3.  **Action-Conditioned World Model (V-JEPA 2-AC):** By adding a small amount of robot data (62 hours), the model learns to predict how its own actions change the environment.
4.  **Zero-Shot Planning:** The model successfully performs robot manipulation tasks (pick and place) in entirely new lab environments without any task-specific training or fine-tuning, demonstrating strong generalization.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Self-Supervised Learning (SSL)
`Self-Supervised Learning` is a method where a model learns from data without needing human-provided labels (like "this is a cat"). Instead, the data itself provides the supervision. For example, the model might hide a part of a video and try to predict what was there based on the surrounding context.

### 3.1.2. Joint-Embedding Predictive Architecture (JEPA)
Proposed by Yann LeCun, `JEPA` is an architecture designed to avoid the pitfalls of generative models. Instead of predicting pixels (which are noisy and hard to get exactly right), a `JEPA` encoder converts an image/video into a **latent embedding** (a vector of numbers representing the meaning). The model then tries to predict the embedding of a missing part of the video from the embedding of the visible part.

### 3.1.3. Vision Transformer (ViT)
The `Vision Transformer` is a neural network architecture that treats an image or video as a sequence of "patches" (like tiles). It uses an **Attention Mechanism** to understand the relationships between different patches. The core of this is the `Self-Attention` formula:

\$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\$

-   $Q$ (Query): What the model is looking for.
-   $K$ (Key): What each part of the data "offers."
-   $V$ (Value): The actual information contained in that part.
-   $d_k$: The dimension of the keys, used for scaling.
-   `softmax`: A function that turns the scores into probabilities (ensuring they sum to 1).

## 3.2. Previous Works
-   **V-JEPA (2024):** The predecessor to this work, which introduced the `Joint-Embedding Predictive Architecture` for video but at a smaller scale.
-   **DINOv2:** A popular image-based SSL model. V-JEPA 2 compares itself to `DINOv2` to show that video-based pre-training is better for understanding motion.
-   **Octo:** A `Vision-Language-Action` (VLA) model that uses imitation learning. Unlike V-JEPA 2, `Octo` predicts actions directly rather than modeling the world's physics.

## 3.3. Technological Evolution
The field has moved from **Supervised Learning** (needing millions of labels) to **Generative SSL** (predicting pixels, e.g., VideoMAE) and now to **Latent SSL** (predicting embeddings, e.g., JEPA). V-JEPA 2 represents the scaling phase of Latent SSL, combining it with robotics for the first time at this magnitude.

---

# 4. Methodology

## 4.1. Principles
The core intuition is that a model that can predict the "missing" parts of a video in a high-level feature space has learned the underlying "laws" of that world. By later conditioning this prediction on an "action," the model becomes a **World Model** that can "imagine" what will happen if it moves its arm in a certain way.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Stage 1: Action-Free Pre-training (V-JEPA 2)
The first goal is to learn a powerful encoder that understands video. The model uses a `mask-denoising` objective. It takes a video $y$, hides some parts (masking), and tries to predict the representation of those parts using the visible parts $x$.

The training objective is to minimize the following loss:

\$
\min_{\theta, \phi, \Delta_y} \| P_{\phi}(\Delta_y, E_{\theta}(x)) - \mathrm{sg}(E_{\bar{\theta}}(y)) \|_1
\$

1.  **Encoder** $E_{\theta}(x)$: A `Vision Transformer` that converts the visible video patches $x$ into feature vectors. $\theta$ represents the weights of this encoder.
2.  **Target Encoder** $E_{\bar{\theta}}(y)$: A "teacher" version of the encoder that processes the full video $y$ to create targets. The weights $\bar{\theta}$ are an **Exponential Moving Average (EMA)** of the main encoder weights $\theta$ to keep training stable.
3.  **Stop-Gradient** $\mathrm{sg}(\cdot)$: This operation ensures that the gradients only flow through the predictor and main encoder, not the teacher, preventing the model from cheating or collapsing.
4.  **Mask Token** $\Delta_y$: A learnable vector that tells the predictor *where* the missing patches were located.
5.  **Predictor** $P_{\phi}$: A smaller transformer that takes the encoder's output and the mask tokens to guess the features of the hidden parts.
6.  **L1 Loss** $\|\cdot\|_1$: The model calculates the absolute difference between its prediction and the actual features produced by the teacher.

### 4.2.2. Stage 2: Action-Conditioned World Model (V-JEPA 2-AC)
After pre-training, the encoder $E$ is frozen. We now train a new predictor $P_{\phi}$ that takes into account robot actions. The model observes a frame $x_t$, a robot state $s_t$ (position of the arm), and an action $a_t$ (how the arm is moving).

The `teacher-forcing loss` is used to train the model to predict the next frame's features $z_{k+1}$:

\$
\mathcal{L}_{\mathrm{teacher-forcing}}(\phi) := \frac{1}{T} \sum_{k=1}^{T} \| P_{\phi}((a_t, s_t, E(x_t))_{t \leq k}) - E(x_{k+1}) \|_1
\$

-   $E(x_t)$: The features of the current video frame.
-   $a_t, s_t$: The action and end-effector state (the "hand" of the robot).
-   $E(x_{k+1})$: The ground-truth features of the *next* frame.

    To ensure the model can think multiple steps ahead, a `rollout loss` is added:

\$
\mathcal{L}_{\mathrm{rollout}}(\phi) := \| P_{\phi}(a_{1:T}, s_1, z_1) - z_{T+1} \|_1
\$

This forces the model to predict step $T+1$ based on its own previous predictions rather than ground truth, which helps prevent error accumulation during planning.

### 4.2.3. Stage 3: Planning via Energy Minimization
To solve a task (e.g., "pick up the cup"), the model is given a **goal image** $x_g$. It calculates the features of that goal `z_g = E(x_g)`. The robot then searches for a sequence of actions $a_{1:T}$ that minimizes an `energy function`:

\$
\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) := \| P(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1
\$

The robot "imagines" different action sequences and picks the one where the final predicted state representation is closest to the goal's representation in latent space. This optimization is performed using the **Cross-Entropy Method (CEM)**, a sampling-based optimization algorithm.

---

# 5. Experimental Setup

## 5.1. Datasets
-   **VideoMix22M (VM22M):** A massive dataset of 22 million samples (1.1M hours) curated from sources like `Something-Something v2`, `Kinetics`, `HowTo100M`, and `YT-Temporal-1B`.
-   **Droid:** 62 hours of robot interaction data (Franka arm) used to train the action-conditioned predictor.
-   **Epic-Kitchens-100 (EK100):** Used for human action anticipation.
-   **VidQA Benchmarks:** `PerceptionTest`, `MVP`, `TempCompass` for testing language understanding of video.

    The researchers provide an example of "curation": they use a cluster-based retrieval to select videos from `YouTube-1B` that match the distribution of high-quality datasets like `Kinetics`, ensuring the model doesn't learn from "noisy" or irrelevant content.

## 5.2. Evaluation Metrics

### 5.2.1. Top-1 Accuracy
1.  **Conceptual Definition:** The percentage of times the model's most likely prediction (highest score) matches the actual correct label.
2.  **Mathematical Formula:**
    \$
    \mathrm{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\hat{y}_i = y_i)
    \$
3.  **Symbol Explanation:** $N$ is total samples; $\hat{y}_i$ is the predicted label; $y_i$ is the true label; $\mathbb{1}(\cdot)$ is the indicator function (1 if true, 0 if false).

### 5.2.2. Mean-Class Recall-at-5
1.  **Conceptual Definition:** For each category, it measures how often the correct label is among the model's top 5 guesses. It averages this across all classes to account for imbalanced data.
2.  **Mathematical Formula:**
    \$
    \mathrm{mR@5} = \frac{1}{C} \sum_{c=1}^{C} \frac{\sum_{i \in S_c} \mathbb{1}(y_i \in \text{Top-5}(\hat{y}_i))}{|S_c|}
    \$
3.  **Symbol Explanation:** $C$ is the number of classes; $S_c$ is the set of samples in class $c$; $\text{Top-5}(\hat{y}_i)$ is the set of the 5 highest-scoring predictions.

## 5.3. Baselines
-   **DINOv2 / SigLIP2:** State-of-the-art image encoders.
-   **Octo:** A leading robot policy model.
-   **Cosmos:** A 7-billion parameter video generation world model from NVIDIA.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The results show that **scaling** is the primary driver of performance. Moving from a 300M parameter model to a 1B parameter model (`ViT-g`) and increasing resolution/duration (up to 64 frames) consistently improved accuracy. In robotics, V-JEPA 2-AC outperformed `Octo` (an imitation model) and `Cosmos` (a generative model), particularly in tasks involving complex object interactions like "Pick-and-Place."

## 6.2. Data Presentation (Tables)
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th colspan="4"></th>
<th colspan="3">Motion Understanding</th>
<th colspan="3">Appearance Understanding</th>
</tr>
<tr>
<th colspan="2">Method</th>
<th>Param.</th>
<th>Avg.</th>
<th>SSv2</th>
<th>Diving-48</th>
<th>Jester</th>
<th>K400</th>
<th>COIN</th>
<th>IN1K</th>
</tr>
</thead>
<tbody>
<tr>
<td>Image Encoders</td>
<td>DINOv2</td>
<td>1.1B</td>
<td>81.1</td>
<td>50.7</td>
<td>82.5</td>
<td>93.4</td>
<td>83.6</td>
<td>90.7</td>
<td>86.1</td>
</tr>
<tr>
<td>Video Encoders</td>
<td>V-JEPA (old)</td>
<td>600M</td>
<td>85.2</td>
<td>74.3</td>
<td>87.9</td>
<td>97.7</td>
<td>84.5</td>
<td>87.1</td>
<td>80.0</td>
</tr>
<tr>
<td>Ours</td>
<td>V-JEPA 2 ViT-g</td>
<td>1B</td>
<td>87.5</td>
<td>75.3</td>
<td>90.1</td>
<td>97.8</td>
<td>86.6</td>
<td>90.7</td>
<td>84.6</td>
</tr>
<tr>
<td>Ours (High Res)</td>
<td>V-JEPA 2 g384</td>
<td>1B</td>
<td>88.2</td>
<td>77.3</td>
<td>90.2</td>
<td>-</td>
<td>87.3</td>
<td>91.1</td>
<td>85.1</td>
</tr>
</tbody>
</table>

The following are the results from Table 2 of the original paper regarding zero-shot robot manipulation:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Reach</th>
<th colspan="2">Grasp</th>
<th colspan="2">Reach w/ Obj.</th>
<th colspan="2">Pick-&-Place</th>
</tr>
<tr>
<th>Cup</th>
<th>Box</th>
<th>Cup</th>
<th>Box</th>
<th>Cup</th>
<th>Box</th>
</tr>
</thead>
<tbody>
<tr>
<td>Octo</td>
<td>100%</td>
<td>15%</td>
<td>0%</td>
<td>15%</td>
<td>70%</td>
<td>15%</td>
<td>10%</td>
</tr>
<tr>
<td><b>V-JEPA 2-AC</b></td>
<td><b>100%</b></td>
<td><b>65%</b></td>
<td><b>25%</b></td>
<td><b>75%</b></td>
<td><b>75%</b></td>
<td><b>80%</b></td>
<td><b>65%</b></td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies
-   **Data Curation:** Training on "curated" YouTube data improved performance by +1.4 points over uncurated data.
-   **Resolution:** Increasing resolution during the "cooldown" phase of training was much more efficient than training at high resolution from the start ($8\times$ speedup).
-   **Predictor Input:** Using both the encoder and the predictor features for action anticipation yielded the best results on the `EK100` dataset.

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
V-JEPA 2 demonstrates that large-scale self-supervised learning on internet video provides a "common sense" foundation for physical world understanding. By learning to predict in a latent space, the model avoids the heavy computation of pixel generation and focuses on the underlying physics. This approach allows robots to plan actions in new environments zero-shot, a major milestone for general-purpose robotics.

## 7.2. Limitations & Future Work
-   **Camera Sensitivity:** The model is sensitive to camera positioning because it implicitly learns action axes from the video; if the camera moves, the "left/right" commands might get confused.
-   **Long Horizon Planning:** Predicting far into the future still suffers from error accumulation.
-   **Task Specification:** Currently, the model requires an "image" of the goal. Future work aims to allow language-based goals (e.g., "Put the cup on the shelf").

## 7.3. Personal Insights & Critique
The most impressive part of this work is the **zero-shot generalization**. Most robot models fail immediately when moved to a new lab with different lighting or backgrounds. V-JEPA 2's ability to handle this suggests that the 1 million hours of internet video successfully taught the model to ignore "background noise" and focus on the objects and the arm.

However, a potential issue is the **CEM optimization** used for planning. Sampling 800 trajectories to find the best one takes 16 seconds per action. While much faster than the generative `Cosmos` model (4 minutes), it is still too slow for high-speed, real-time robotics. Future iterations might need a "policy" that distills the world model's knowledge into a faster, reactive neural network.