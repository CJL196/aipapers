# 1. Bibliographic Information

## 1.1. Title
LAOF: Robust Latent Action Learning with Optical Flow Constraints

## 1.2. Authors
Xizhou Bu, Jiexi Lyu, Fulei Sun, Ruichen Yang (Fudan University); Zhiqiang Ma (Northwestern Polytechnical University); Wei Li (Fudan University).

## 1.3. Journal/Conference
This paper was published as a preprint on arXiv (2025). Based on the formatting and content, it is targeted toward top-tier robotics and machine learning conferences such as **ICLR**, **ICRA**, or **CVPR**.

## 1.4. Publication Year
2025 (Published at UTC: 2025-11-20T14:26:49.000Z).

## 1.5. Abstract
The paper addresses the challenge of pre-training scalable embodied foundation models from large-scale videos. Existing methods for learning latent actions (representations of motion without explicit action labels) often fail due to action-irrelevant "distractors" in the background. The authors propose **LAOF** (Latent Action learning with Optical Flow), a framework that uses the agent's optical flow—pixel-level motion between frames—as a pseudo-supervision signal. This constraints the model to learn representations focused on the agent's actual movement. Experiments on robotics (LIBERO) and reinforcement learning (PROCGEN) benchmarks show that LAOF improves the quality of latent actions, providing stable training even when action labels are scarce or non-existent.

## 1.6. Original Source Link
- **Official ArXiv Link:** [https://arxiv.org/abs/2511.16407](https://arxiv.org/abs/2511.16407)
- **PDF Link:** [https://arxiv.org/pdf/2511.16407v1](https://arxiv.org/pdf/2511.16407v1)

  ---

# 2. Executive Summary

## 2.1. Background & Motivation
The current trend in robotics is to build "Foundation Models"—large neural networks trained on massive datasets that can perform many tasks. However, unlike text or images, robot data (videos paired with actions) is hard to collect. To solve this, researchers use **Latent Action Models (LAMs)**, which learn to predict "hidden" actions from raw videos without needing manual action labels.

**The Core Problem:**
Existing LAMs (like the `LAPO` framework) assume that any change between two video frames is caused by the robot's action. In the real world, this is rarely true. Moving backgrounds, changing lights, or other objects (distractors) can trick the model into thinking they are part of the action. This leads to "entangled" representations where the model confuses visual appearance with actual physical motion.

**The Entry Point:**
The authors observe that **Optical Flow** (the pattern of apparent motion of objects in a visual scene) naturally highlights moving objects and suppresses static backgrounds. By forcing the model to reconstruct the optical flow of the agent, they can "anchor" the latent actions to actual physical movement.

## 2.2. Main Contributions / Findings
1.  **Proposed LAOF:** A new framework that integrates an **Optical Flow constraint** into latent action learning. It uses a pre-trained optical flow model (like `RAFT`) to generate pseudo-labels for training.
2.  **Robustness to Distractors:** The method effectively ignores background noise, making the learned actions much "cleaner" and more useful for downstream tasks.
3.  **Label Efficiency:** Even with **0%** action labels, LAOF matches the performance of previous state-of-the-art methods that use **1%** action supervision. It remains effective and beneficial up to a **10%** label ratio.
4.  **SOTA Performance:** Demonstrated significant improvements in success rates for robot manipulation (LIBERO) and episodic returns in reinforcement learning games (PROCGEN).

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

*   **Latent Action:** In robotics, an "action" is usually a vector of motor commands (e.g., "rotate joint A by 5 degrees"). A *latent* action is a mathematical representation (a vector of numbers) learned by a model to represent the difference between two video frames, without knowing the actual motor commands.
*   **Inverse Dynamics Model (IDM):** A model that takes two consecutive frames ($O_t$ and $O_{t+1}$) and tries to guess what action was taken to get from the first to the second.
*   **Forward Dynamics Model (FDM):** A model that takes a current frame ($O_t$) and an action (real or latent) and predicts what the *next* frame ($O_{t+1}$) will look like.
*   **Optical Flow:** A technique in computer vision that calculates how much each pixel moved between two frames. It results in a "flow field" where each pixel has a 2D vector `(u, v)` representing its horizontal and vertical displacement.
*   **DINOv2:** A powerful "foundation" visual encoder from Meta. It turns images into high-level feature vectors that are very good at capturing object shapes and parts without being specifically trained for a single task.

## 3.2. Previous Works
The paper builds primarily on the **LAPO (Latent Action Policies)** paradigm.
*   **LAPO (2024):** Introduced the idea of training an IDM and FDM together using a reconstruction objective. If the FDM can successfully predict the next frame using the action provided by the IDM, the IDM must have learned a useful representation of motion.
*   **LAOM (2025):** Noted that LAPO fails when there are distractors (moving backgrounds). They suggested adding a small amount of human-labeled actions to "guide" the latent actions.

## 3.3. Technological Evolution
1.  **Stage 1: Supervised Learning.** Robots were trained on pairs of (Image, Action). This is accurate but doesn't scale because labels are expensive.
2.  **Stage 2: Unsupervised Latent Action (LAPO).** Models learned from raw video. This scales perfectly but is easily confused by background noise.
3.  **Stage 3: Pseudo-Supervised Latent Action (LAOF - This Paper).** Uses "natural" labels like Optical Flow to get the accuracy of supervised learning with the scalability of unsupervised learning.

    ---

# 4. Methodology

## 4.1. Principles
The core idea is to treat **Optical Flow** as a "ground truth" for motion. The model is forced to predict not just the next state of the world, but specifically the pixel-level movement that occurred. This ensures the latent action $z$ captures the "how" of the movement, not just the "what" of the visual change.

## 4.2. Core Methodology In-depth

The following figure (Figure 1 from the original paper) shows the system architecture:

![](images/1.jpg)

### 4.2.1. Feature Encoding and State Space
The process begins by converting raw observations (images) into a compact feature space. Given consecutive observations $(o_t, o_{t+1})$ at time $t$ and $t+1$, a pre-trained **DINOv2** visual encoder processes them into states $(s_t, s_{t+1}) \in S = \mathbb{R}^d$. Simultaneously, the optical flow $f_{rgb,t}$ is generated and encoded into the same state space $f_t \in S$.

### 4.2.2. The Inverse Dynamics Model (IDM)
The IDM, implemented as a spatial-temporal transformer, acts as the "encoder" for the action. It observes the transition from the current state to the next state and produces a latent action $z_t$:
$z_t \sim p_{IDM}(z_t | s_t, s_{t+1})$
Where $z_t \in \mathcal{Z} = \mathbb{R}^k$ is the latent representation of the action.

### 4.2.3. The Forward Dynamics Model (FDM) and Reconstruction
To ensure $z_t$ is useful, the FDM must use it to predict the future state $\hat{s}_{t+1}$:
$\hat{s}_{t+1} \sim p_{FDM}(\hat{s}_{t+1} | s_t, z_t)$
The model is optimized by minimizing the **Next-State Reconstruction Loss**:
\$
\mathcal{L}_{reconstruction}(t) := \| \hat{s}_{t+1} - s_{t+1} \|_2
\$
This formula calculates the Euclidean distance (L2 norm) between the predicted feature vector and the actual feature vector of the next frame.

### 4.2.4. The Optical Flow Constraint (The Core Innovation)
To prevent the IDM from learning "shortcuts" (like just memorizing background changes), the authors introduce a **Flow Decoder** $d_{flow}$. This decoder takes the latent action $z_t$ and attempts to reconstruct the optical flow $\hat{f}_t$:
$\hat{f}_t = d_{flow}(z_t)$
The **Optical Flow Constraint Loss** is defined as:
\$
\mathcal{L}_{flow}(t) := \| \hat{f}_t - \bar{f}_t \|_2
\$
Where $\bar{f}_t$ is the "pseudo-label" flow generated by a pre-trained model (RAFT). This forces the latent action $z$ to align with the actual physical motion occurring in the pixels.

### 4.2.5. Combined Pre-training Objective
The total loss for training the base LAOF model is the sum of the reconstruction and the flow constraints:
\$
\mathcal{L}_{pretrain} = \mathcal{L}_{reconstruction} + \mathcal{L}_{flow}
\$

### 4.2.6. Learning with Sparse Action Supervision (LAOF-Action)
In scenarios where a small number of real physical actions $a_t$ are available, the model can be further refined. The authors introduce a balancing coefficient $\lambda$ to combine pseudo-supervision (flow) and real supervision (actions):
\$
\mathcal{L}_{pretrain} = \mathcal{L}_{reconstruction} + (1 - \lambda) \cdot \mathcal{L}_{flow} + \lambda \cdot \mathcal{L}_{action}
\$
Where $\mathcal{L}_{action} := \| d_{action}(\hat{a}_t | z_t) - a_t \|_2$ and $\lambda = \frac{M}{N+M}$ (the ratio of labeled data $M$ to total data $N+M$).

### 4.2.7. Distillation for Deployment
Since a robot cannot see the future frame $s_{t+1}$ during real-time operation, the model "distills" the knowledge from the IDM into a **Policy** $\pi$. The policy learns to predict the same latent action $z_t$ using only the current state $s_t$ and a language instruction $l_t$:
\$
\mathcal{L}_{distillation} := \| \pi(\hat{z}_t | s_t, l_t) - z_t \|_2
\$

---

# 5. Experimental Setup

## 5.1. Datasets
The authors used two distinct benchmarks to test the model's versatility:

1.  **LIBERO:** A robot manipulation benchmark. It features a robotic arm in a kitchen-like environment performing tasks based on language instructions (e.g., "pick up the black bowl").
    *   **Scale:** 4 suites (Spatial, Object, Goal, Long), each with 10 tasks.
    *   **Data:** 50 human-teleoperated demonstrations per task.
2.  **PROCGEN:** A suite of 16 procedurally generated games designed to test reinforcement learning.
    *   **Environments used:** `BIGFISH`, `CHASER`, `LEAPER`, `HEIST`.
    *   **Complexity:** These games have dynamic backgrounds and stochastic (random) elements, providing a heavy test for "distractor" robustness.

## 5.2. Evaluation Metrics

1.  **Mean Squared Error (MSE):** Used for continuous action tasks (like robot arm joints).
    *   **Conceptual Definition:** Quantifies the average squared difference between the predicted action vector and the ground-truth action vector. Lower is better.
    *   **Mathematical Formula:**
        \$
        \mathbf{MSE} = \frac{1}{M} \sum_{i=1}^{M} \| \hat{a}_i - a_i \|_2
        \$
    *   **Symbol Explanation:** $M$ is the number of samples, $\hat{a}_i$ is the predicted action, $a_i$ is the ground-truth action.

2.  **Success Rate (Succ.):** Used for robot tasks.
    *   **Conceptual Definition:** The percentage of trials where the robot successfully completed the instructed task within a time limit. Higher is better.

3.  **Classification Accuracy (Acc.):** Used for discrete actions (like "move left," "jump" in games).
    *   **Conceptual Definition:** The ratio of correctly predicted actions to the total number of actions. Higher is better.
    *   **Mathematical Formula:**
        \$
        \mathrm{Acc} = \frac{1}{M} \sum_{i=1}^{M} \mathbb{1} [ \hat{a}_i = a_i ]
        \$
    *   **Symbol Explanation:** $\mathbb{1} [ \cdot ]$ is the indicator function (1 if correct, 0 if wrong).

## 5.3. Baselines
*   **LAPO:** The standard unsupervised baseline (Next-state reconstruction only).
*   **CoMo:** A variant that uses "inter-frame differences" (frame $t+1$ minus frame $t$) instead of raw frames to focus on change.
*   **LAOM-Action:** A recent state-of-the-art method that uses a small amount of action labels (1%) to supervise the latent space.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The results indicate that LAOF significantly improves both the "cleanness" of learned actions (lower MSE) and the final task performance (Success/Return).

**Key Observation:** On the LIBERO benchmark, **LAOF (Unsupervised)** achieved a success rate of **+4.2%** over the baseline. Remarkably, when adding just 1% action labels (**LAOF-Action**), the improvement jumped to **+11.5%**.

The following are the results from Table 1 of the original paper, comparing LAOF to other methods on the LIBERO benchmark:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">SPATIAL</th>
<th colspan="2">OBJECT</th>
<th colspan="2">GOAL</th>
<th colspan="2">LONG</th>
<th colspan="2">Avg. Impr.</th>
</tr>
<tr>
<th>MSE (↓)</th>
<th>Succ. (↑)</th>
<th>MSE (↓)</th>
<th>Succ. (↑)</th>
<th>MSE (↓)</th>
<th>Succ. (↑)</th>
<th>MSE (↓)</th>
<th>Succ. (↑)</th>
<th>MSE (↓)</th>
<th>Succ. (↑)</th>
</tr>
</thead>
<tbody>
<tr>
<td>LAPO [32]</td>
<td>0.162</td>
<td>80.4 ± 1.7</td>
<td>0.139</td>
<td>81.2 ± 2.4</td>
<td>0.219</td>
<td>84.0 ± 2.2</td>
<td>0.154</td>
<td>44.7 ± 1.6</td>
<td>-0.000</td>
<td>+0.0</td>
</tr>
<tr>
<td>CoMo [40]</td>
<td>0.181</td>
<td>74.1 ± 1.8</td>
<td>0.125</td>
<td>87.6 ± 1.3</td>
<td>0.221</td>
<td>80.8 ± 2.7</td>
<td>0.153</td>
<td>49.9 ± 1.8</td>
<td>+0.02</td>
<td>+0.5</td>
</tr>
<tr>
<td>LAOF (Ours)</td>
<td>0.111</td>
<td>82.5 ± 2.3</td>
<td>0.082</td>
<td>85.3 ± 1.4</td>
<td>0.118</td>
<td>87.2 ± 2.2</td>
<td>0.088</td>
<td>52.0 ± 1.7</td>
<td>-0.069</td>
<td>+4.2</td>
</tr>
<tr>
<td>LAOM-Action [27] (1% Label)</td>
<td>0.108</td>
<td>86.0 ± 2.3</td>
<td>0.090</td>
<td>91.1 ± 1.5</td>
<td>0.127</td>
<td>86.3 ± 1.7</td>
<td>0.086</td>
<td>61.6 ± 2.3</td>
<td>-0.066</td>
<td>+8.7</td>
</tr>
<tr>
<td>LAOF-Action (Ours, 1% Label)</td>
<td>0.076</td>
<td>88.2 ± 1.5</td>
<td>0.064</td>
<td>95.9 ± 1.3</td>
<td>0.081</td>
<td>88.6 ± 1.6</td>
<td>0.068</td>
<td>63.7 ± 1.9</td>
<td>-0.096</td>
<td>+11.5</td>
</tr>
</tbody>
</table>

The following are the results from Table 2 of the original paper, showing performance on the PROCGEN (Game) benchmark:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">BIGFISH</th>
<th colspan="2">CHASER</th>
<th colspan="2">LEAPER</th>
<th colspan="2">HEIST</th>
<th colspan="2">Avg. Impr.</th>
</tr>
<tr>
<th>Acc. (↑)</th>
<th>Return (↑)</th>
<th>Acc. (↑)</th>
<th>Return (↑)</th>
<th>Acc. (↑)</th>
<th>Return (↑)</th>
<th>Acc. (↑)</th>
<th>Return (↑)</th>
<th>Acc. (↑)</th>
<th>Return (↑)</th>
</tr>
</thead>
<tbody>
<tr>
<td>LAPO [32]</td>
<td>80.98</td>
<td>0.55</td>
<td>26.87</td>
<td>0.09</td>
<td>40.09</td>
<td>0.55</td>
<td>72.23</td>
<td>0.72</td>
<td>+0.00</td>
<td>+0.00</td>
</tr>
<tr>
<td>LAOF (Ours)</td>
<td>83.71</td>
<td>0.76</td>
<td>53.83</td>
<td>0.39</td>
<td>53.23</td>
<td>0.74</td>
<td>94.19</td>
<td>0.88</td>
<td>+16.20</td>
<td>+0.16</td>
</tr>
<tr>
<td>LAOF-Action (1% Label)</td>
<td>84.13</td>
<td>0.80</td>
<td>62.75</td>
<td>0.51</td>
<td>57.64</td>
<td>0.79</td>
<td>98.57</td>
<td>0.91</td>
<td>+20.73</td>
<td>+0.22</td>
</tr>
</tbody>
</table>

## 6.2. Stability and Overfitting
As can be seen from the results in Figure 5 (below), LAOF-Action is much more stable than previous methods. In the `CHASER` environment, for example, the standard `LAOM-Action` (1% label) suffers from extreme variance (the shaded area), whereas LAOF-Action remains consistent. This suggests that optical flow prevents the model from "overfitting" to specific visual quirks of the limited labeled data.

![Figure 5. Comparison of stability and overfitting among different methods, where solid lines represent unsupervised methods and dashed lines represent action-supervised methods. LAOM-Action and LAOF-Action are evaluated at a $1 \\%$ action ratio.](images/4.jpg)
*Figure 5. Comparison of stability and overfitting among different methods, where solid lines represent unsupervised methods and dashed lines represent action-supervised methods. LAOM-Action and LAOF-Action are evaluated at a $1 \%$ action ratio.*

## 6.3. Ablation Studies
The authors compared different ways to use optical flow:
*   **LAOF-Only:** Learning actions *only* using flow (no reconstruction of the next frame). Performance dropped, proving that reconstruction is still a necessary "structural" task.
*   **LAOF-AE:** Simple autoencoding of flow. It performed surprisingly well, showing that flow itself is a very strong signal for learning motion.
*   **Optimal Architecture:** The best results came from having a **dedicated decoder** specifically for flow, rather than mixing flow prediction into the FDM.

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that **Optical Flow** is a superior "prior" for latent action learning. By forcing the model to reproduce the pixel-level displacement of the agent, LAOF solves the long-standing problem of environmental distractors in unsupervised learning. The method is highly label-efficient, stable, and outperforms both unsupervised and semi-supervised baselines across diverse tasks in robotics and gaming.

## 7.2. Limitations & Future Work
*   **Reliance on Flow Models:** LAOF depends on the quality of the optical flow pseudo-labels. If the underlying flow model (like RAFT) fails in a specific domain, LAOF will also struggle.
*   **Object Segmentation:** For complex dynamic scenes, the authors used `LangSAM` to find the robot's pixels. This text-to-segmentation step can be imprecise.
*   **Camera Configuration:** The current method is optimized for "eye-off-hand" (stationary camera). Future work is needed to adapt this to "eye-in-hand" (camera on the robot's wrist), where the background moves because the camera is moving.

## 7.3. Personal Insights & Critique
**Inspiration:** The idea of using "low-level" physics (optical flow) to guide "high-level" representation learning is very clever. It bridges the gap between pure computer vision and physical embodiment.

**Critique:** While the 1% label results are impressive, the paper notes that the benefit of optical flow disappears after the action label ratio exceeds 10%. This suggests that while optical flow is a great "starter" signal, it lacks the nuance of true human action labels once enough of them are available. Furthermore, the conversion of optical flow to RGB format to fit DINOv2 is a clever engineering hack, but it might lose some mathematical precision inherent in the raw vector fields. One might wonder if a more direct multi-modal fusion would work even better.