# 1. Bibliographic Information

## 1.1. Title
CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning

## 1.2. Authors
Jiange Yang (Nanjing University, Shanghai AI Lab), Yansong Shi, Haoyi Zhu, Mingyu Liu, Kaijing Ma, Yating Wang, Gangshan Wu, Tong He, Limin Wang.

## 1.3. Journal/Conference
This paper was published as a preprint on arXiv (2025-05-22). Given the authors' affiliations and the quality of the work, it is likely targeted at a top-tier computer vision or robotics conference such as CVPR, ICCV, or CoRL.

## 1.4. Publication Year
2025

## 1.5. Abstract
The paper addresses the challenge of scaling robot learning by utilizing vast amounts of action-less Internet videos. While previous methods often used discrete latent actions (represented as tokens), they suffered from information loss and struggled with fine-grained dynamics. The authors propose `CoMo` (Continuous Motion), a framework to learn informative continuous motion representations. To prevent the model from simply memorizing the background (model collapse), they introduce an `early temporal feature difference` mechanism. They also propose two new evaluation metrics: `LP-MSE` (Linear Probing Mean Squared Error) and `S-PCFC` (Similarity between Past-to-Current and Future-to-Current motion). `CoMo` demonstrates strong zero-shot generalization, allowing it to generate "pseudo-actions" for human and internet videos, which can then be used to train robust robot policies.

## 1.6. Original Source Link
- **PDF Link:** [https://arxiv.org/pdf/2505.17006v1](https://arxiv.org/pdf/2505.17006v1)
- **Status:** Preprint (under review).

  ---

# 2. Executive Summary

## 2.1. Background & Motivation
Robot learning is currently bottlenecked by the "data scarcity" problem. Unlike Large Language Models (LLMs) that have the entire internet's text to learn from, robots require specialized data consisting of paired observations (images) and actions (joint movements). Collecting this data is slow and expensive.

A promising solution is to use the millions of videos on the internet (human cooking, DIY, etc.). However, these videos lack action labels (we don't know the exact torque or velocity of the human's joints). Recent works have tried to learn "latent actions" (hidden representations of movement) from these videos. Most existing methods use `discrete` tokens (like words in a dictionary), but the real world is `continuous`. 

The core challenge in learning continuous motion is **model collapse** or **shortcut learning**: if a model is asked to predict the next frame, it might just learn to encode the visual appearance of the future frame rather than the actual "motion" required to get there.

## 2.2. Main Contributions / Findings
1.  **Continuous Latent Motion:** Proposes `CoMo`, a framework that learns continuous rather than discrete motion representations, preserving fine-grained details.
2.  **Early Temporal Difference:** A mechanism that subtracts features of the current frame from the future frame to highlight motion and suppress static background noise, preventing shortcut learning.
3.  **New Evaluation Metrics:** Introduces `LP-MSE` and `S-PCFC` to evaluate motion quality affordably without needing expensive real-robot trials for every iteration.
4.  **Scalable Joint Training:** Demonstrates that robots can be trained using a mix of real robot data and "pseudo-labeled" internet videos, significantly improving performance and generalization.
5.  **Zero-Shot Capability:** The learned motion model can be applied to completely unseen video domains (like human hands) without any additional training.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Inverse Dynamics Model (IDM)
An `Inverse Dynamics Model` is a neural network that takes two consecutive visual observations (images) as input—the current state $O_t$ and the future state $O_{t+n}$—and attempts to predict the action $a_t$ (or latent motion $z_t$) that caused the transition between them. 
*   **Intuition:** "If I see the cup on the table at time 1, and the cup in the air at time 2, what movement must have happened?"

### 3.1.2. Forward Dynamics Model (FDM)
A `Forward Dynamics Model` (also called a World Model) predicts the next observation $\hat{O}_{t+n}$ given the current observation $O_t$ and an action (or latent motion $z_t$).
*   **Intuition:** "If the cup is on the table and I lift my arm, what will the next image look like?"

### 3.1.3. Vector Quantization (VQ)
`Vector Quantization` is a technique used in models like `VQ-VAE` to force continuous neural representations into a finite "codebook" of discrete vectors (tokens). 
*   **Problem in Robotics:** While VQ prevents the model from just copying the future frame (because the "bottleneck" is so tight), it discards subtle nuances of movement (e.g., how slowly a hand rotates), which are critical for precise manipulation.

### 3.1.4. Information Bottleneck Principle
The `Information Bottleneck` theory suggests that an optimal representation should contain all the information necessary to predict the output (task-relevant info) while discarding all irrelevant "noise" (like the color of the wall in a robot task).

## 3.2. Previous Works
*   **LAPA & Moto-GPT:** These are recent frameworks that learn discrete latent actions. They use an encoder-decoder structure where the encoder is the IDM and the decoder is the FDM. They rely on discrete tokens to avoid model collapse.
*   **ATM (Any-point Trajectory Modeling):** Instead of latent actions, this predicts the trajectories of specific points (pixels) in the image to guide the robot.
*   **Diffusion Policy:** A state-of-the-art method for robot control that uses a `Diffusion Model` (similar to Stable Diffusion) to generate smooth, continuous action sequences.

## 3.3. Differentiation Analysis
Unlike previous works that rely on `discrete` tokens to avoid shortcut learning, `CoMo` uses **continuous** representations but introduces a architectural "guardrail" (the temporal feature difference) to ensure the model focuses on motion rather than static appearance.

---

# 4. Methodology

## 4.1. Principles
The core intuition of `CoMo` is that motion is the *difference* between two states. By explicitly feeding the network the difference between image features, we force it to look at what changed (the movement) rather than what stayed the same (the background).

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Stage 1: Learning the Continuous Latent Motion
The following figure (Figure 1 from the original paper) shows the system architecture:

![Figure 1: The CoMo framework. In the first stage, we self-supervisely learn inter-frame latent motion representations from Internet videos. In the second stage, we directly utilize the IDM trained in the first stage to extract pseudo action labels for action-less video data, ensuring joint learning of continuous robot action data and action-less video data under a unified policy architecture.](images/1.jpg)
*该图像是CoMo框架的示意图。在第一阶段，我们自监督学习互联网视频中的帧间潜在运动表示；在第二阶段，利用第一阶段训练的IDM提取无动作视频数据的伪动作标签，确保在统一政策架构下联合学习连续机器人动作数据和无动作视频数据。*

#### Step 1: Feature Extraction
Given a current frame $O_t$ and a future frame $O_{t+n}$, we use a shared `Vision Transformer` (specifically a `ViT-Large` pretrained with `MAE`) to extract high-level feature maps. Let these features be $F_t$ and $F_{t+n}$.
$F = \mathrm{ViT}(O)$
Where $O$ is the input image and $F$ is the resulting token-level feature representation.

#### Step 2: Early Temporal Feature Difference (The "Motion-Enhanced" IDM)
To suppress the static background and highlight moving parts, we compute the difference $D_t$ between the future and current features:
$D_t = F_{t+n} - F_t$
Crucially, to prevent shortcut learning (where the model just looks at the future frame's appearance), the authors **explicitly remove** $F_{t+n}$ from the encoder's input. The input to the `Motion Q-former` (the encoder) is the concatenation of the current feature and the difference:
$\mathrm{Input}_{IDM} = [F_t, D_t]$
This ensures that the encoder "sees" the starting point and the *change*, but not the final image itself.

#### Step 3: Latent Motion Extraction ($Z_t$)
A set of learnable `query embeddings` (placeholders) interacts with $[F_t, D_t]$ through Transformer layers. The output of these queries becomes our continuous latent motion representation $Z_t$.
$Z_t = \mathrm{MotionQFormer}([F_t, D_t])$

#### Step 4: Reconstruction via Forward Dynamics Model (FDM)
The FDM acts as a decoder. It takes the current observation $O_t$ and the latent motion $Z_t$ and tries to reconstruct the future frame $\hat{O}_{t+n}$.
1.  The image $O_t$ is turned into patch embeddings.
2.  The motion $Z_t$ is pooled and added to these embeddings: $E(O_t, Z_t) = \mathrm{PatchEmbed}(O_t) + \mathrm{Pool}(Z_t)$.
3.  A Transformer and a Convolutional decoder generate the predicted image $\hat{O}_{t+n}$.

#### Step 5: Training Objectives
The model is trained to minimize the difference between the predicted future frame $\hat{O}_{t+n}$ and the ground truth $O_{t+n}$ using a combination of Pixel-level loss ($L_2$) and Perceptual loss (which focuses on high-level features).

### 4.2.2. Stage 2: Joint Policy Learning
Once the IDM is trained, it can be used to label any video.
1.  **Robot Data:** We have $(O_t, a_t)$, where $a_t$ is the real robot command.
2.  **Internet Video:** we have $(O_t)$, but no $a_t$. We use the IDM to get $z_t = \mathrm{IDM}(O_t, O_{t+n})$.
3.  **Unified Training:** We train a policy (like a `Diffusion Policy` or `VLA`) to predict *both* $a_t$ and $z_t$. Because both are continuous, the model can learn shared "principles of movement" from both datasets simultaneously.

    ---

# 5. Experimental Setup

## 5.1. Datasets
*   **LIBERO:** A benchmark for lifelong robot learning with 130 tasks (e.g., picking objects, opening drawers).
*   **CALVIN:** A long-horizon benchmark using a Franka arm in a simulated kitchen.
*   **Internet/Human Data:** 120,000 videos sampled from `SAM-V` (segmented videos), `EgoVid` (first-person human videos), and `Droid` (diverse robot data).
*   **Real-World:** A physical Franka Emika Research 3 robot arm.

## 5.2. Evaluation Metrics

### 5.2.1. Linear Probing MSE (LP-MSE)
1.  **Conceptual Definition:** Measures how much "action-relevant" information is inside the latent vector. If a simple linear layer can predict the real robot action from the latent vector, then the latent vector is a good representation of motion.
2.  **Mathematical Formula:**
    \$
    \begin{array} { c } { \hat { a } _ { t } = \mathbf { MLP } ( z _ { t } ) , } \\ { \mathbf { LP } \mathbf { - M S E } ( t ) = \mathbf { MSE } ( a _ { t } , \hat { a } _ { t } ) . } \end{array}
    \$
3.  **Symbol Explanation:** $z_t$ is the latent motion vector; $\mathbf{MLP}$ is a single linear layer; $a_t$ is the ground truth robot action; $\hat{a}_t$ is the predicted action. **Lower is better.**

### 5.2.2. Cosine Similarity (S-PCFC)
1.  **Conceptual Definition:** This detects "shortcut learning." It compares the motion representation of a forward transition ($t$ to $t+n$) with a backward transition ($t+n$ to $t$). If they are highly similar, it means the model is just looking at static background stuff that doesn't change when time is reversed.
2.  **Mathematical Formula:**
    \$
    \operatorname { S - P C F C } ( t ) = { \frac { z { \big ( } o _ { t - n } , o _ { t } { \big ) } ^ { \top } z { \big ( } o _ { t + n } , o _ { t } { \big ) } } { \left\| z { \big ( } o _ { t - n } , o _ { t } { \big ) } \right\| _ { 2 } \left\| z { \big ( } o _ { t + n } , o _ { t } { \big ) } \right\| _ { 2 } } }
    \$
3.  **Symbol Explanation:** $z(o_1, o_2)$ is the latent vector derived from those two frames. The formula calculates the cosine of the angle between the two vectors. **Lower is better** (indicating directional sensitivity).

## 5.3. Baselines
*   **DP (Diffusion Policy):** Standard imitation learning on robot data only.
*   **Pre-VQ:** A version where vectors are forced toward discrete points (similar to LAPA).
*   **GR2-like:** Using raw future frame features as motion (prone to shortcut learning).

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
*   **Scalability:** Adding internet video data via `CoMo` increased the success rate on LIBERO from **70.4% to 80.8%**.
*   **Continuous vs. Discrete:** `CoMo` outperformed `Pre-VQ` (discrete) by a wide margin (80.8% vs. 73.6%), proving that discrete tokens lose too much detail for complex tasks.
*   **Shortcut Prevention:** The `S-PCFC` metric showed that without the feature difference mechanism, the model had a similarity of **0.989** (almost total collapse), whereas `CoMo` achieved **0.901** (much better directional capture).

## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper, showing the ablation study on LIBERO:

<table>
<thead>
<tr>
<th>Suites</th>
<th>Metric</th>
<th>O2-Fea</th>
<th>w/o. VQ</th>
<th>Pre-VQ</th>
<th>RGB-Diff</th>
<th>Fea-Diff (Ours)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Spatial</td>
<td>Success Rate ↑</td>
<td>81.0±3.0</td>
<td>81.7±1.2</td>
<td>76.0±0.8</td>
<td>82.7±4.1</td>
<td>80.3±1.2</td>
</tr>
<tr>
<td>LP-MSE ↓</td>
<td>1.208</td>
<td>1.189</td>
<td>3.055</td>
<td>0.891</td>
<td>0.881</td>
</tr>
<tr>
<td>S-PCFC ↓</td>
<td>1.000</td>
<td>0.988</td>
<td>0.821</td>
<td>0.786</td>
<td>0.892</td>
</tr>
<tr>
<td rowspan="3">Object</td>
<td>Success Rate ↑</td>
<td>95.7±0.5</td>
<td>93.0±2.2</td>
<td>89.3±1.2</td>
<td>92.3±1.2</td>
<td>95.0±0.0</td>
</tr>
<tr>
<td>LP-MSE ↓</td>
<td>0.896</td>
<td>0.865</td>
<td>2.363</td>
<td>0.604</td>
<td>0.662</td>
</tr>
<tr>
<td>S-PCFC ↓</td>
<td>1.000</td>
<td>0.992</td>
<td>0.810</td>
<td>0.810</td>
<td>0.902</td>
</tr>
<tr>
<td colspan="2"><b>Avg. Success Rate ↑</b></td>
<td>75.7</td>
<td>77.7</td>
<td>73.6</td>
<td>79.8</td>
<td><b>80.8</b></td>
</tr>
</tbody>
</table>

*(Note: Table truncated for clarity; Avg. values represent the mean across all 4 suites: Spatial, Object, Goal, Long).*

## 6.3. Ablation Studies / Parameter Analysis
As seen in Figure 3 of the paper:
*   **Dimension Scaling:** The authors tested different sizes for the motion vector $Z_t$. They found that **128 dimensions** provided the best balance. 
*   Too small (e.g., 16): Not enough info (high LP-MSE).
*   Too large (e.g., 256): The model starts capturing "visual noise" rather than motion (high S-PCFC).

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
`CoMo` successfully demonstrates that **continuous** latent motion is superior to discrete tokens for robot learning from videos. By using an early feature difference and removing future frames from the encoder, it prevents the model from "cheating" and ensures it learns real dynamics. This allows robots to "watch" internet videos of humans or other robots and translate that into useful knowledge for their own tasks.

## 7.2. Limitations & Future Work
*   **The "Action Gap":** There is still a performance gap between having real action data and using `CoMo` pseudo-actions (89.2% for 5x robot data vs. 80.8% for robot + video data). 
*   **Temporal Sensitivity:** The authors suggest that adding more explicit temporal supervision (e.g., video speed or frame ordering constraints) could further improve the motion representation.
*   **Dynamics:** Currently, the model focuses on visual change. Future work could incorporate physical constraints (gravity, friction) into the latent space.

## 7.3. Personal Insights & Critique
This paper is a significant step forward because it challenges the "discretization dogma" currently prevalent in VLA models (like RT-2 or Moto-GPT). 
**Strengths:**
*   The `S-PCFC` metric is an ingenious way to measure model collapse without needing a robot.
*   The framework is "architecture agnostic," meaning it works with both Diffusion and Autoregressive (GPT-like) policies.

**Potential Issues:**
*   **Camera Motion:** If the video camera is moving (shaky hands), the "feature difference" $D_t$ will be dominated by camera movement rather than the object's movement. The paper doesn't deeply discuss how they handle non-static camera backgrounds in internet videos, which is a major challenge for this approach.