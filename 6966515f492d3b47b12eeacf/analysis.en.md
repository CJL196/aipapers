# 1. Bibliographic Information

## 1.1. Title
Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model

## 1.2. Authors
Junshu Tang, Jiacheng Liu, Jiaqi Li, Longhuang Wu, Haoyu Yang, Penghao Zhao, Siruis Gong, Xiang Yuan, Shuai Shao, Qinglin Lu.
**Affiliations:** Tencent Hunyuan. The authors are researchers from Tencent’s flagship AI team, specializing in generative models and computer vision.

## 1.3. Journal/Conference
This paper is a technical report/preprint hosted on ArXiv (arXiv:2511.23429). It represents cutting-edge industry research in generative AI for gaming, a rapidly evolving sub-field of computer vision.

## 1.4. Publication Year
Published at (UTC): 2025-11-28.

## 1.5. Abstract
The paper introduces `Hunyuan-GameCraft-2`, a generative world model designed to create interactive game environments. Unlike previous models that rely on rigid keyboard schemas, this model allows users to control game video content through natural language instructions (e.g., "open the door," "trigger an explosion"). The authors define the concept of `Interactive Video Data` and create an automated pipeline to generate such data at scale. Built on a 14B `Mixture-of-Experts (MoE)` foundation, the model features a text-driven interaction injection mechanism and an autoregressive distillation strategy to achieve real-time (16 FPS) long-horizon video generation. They also propose `InterBench` to evaluate interaction performance across dimensions like physical plausibility and causal coherence.

## 1.6. Original Source Link
- **Official ArXiv Link:** [https://arxiv.org/abs/2511.23429](https://arxiv.org/abs/2511.23429)
- **PDF Link:** [https://arxiv.org/pdf/2511.23429v1](https://arxiv.org/pdf/2511.23429v1)

  ---

# 2. Executive Summary

## 2.1. Background & Motivation
Traditional game world models have evolved from synthesizing static scenes to creating dynamic videos. However, two major hurdles remain:
1.  **Rigid Interaction:** Most models only respond to fixed inputs like W/A/S/D keys or mouse movements, lacking the ability to understand complex semantic instructions (e.g., "draw a sword").
2.  **Data Scarcity:** High-quality "interactive" data—where an action clearly causes a state change—is difficult to find or expensive to annotate manually.

    The core motivation is to move from "what the world looks like" to "how we interact with it," enabling a truly "playable" world that follows free-form human intent.

## 2.2. Main Contributions / Findings
- **Definition of Interactive Video Data:** Formally defines data that captures "actions executed by an agent triggering state transitions with clear causal relationships."
- **Automated Data Pipeline:** Developed a method to transform unstructured gameplay and text-video pairs into causally aligned interactive datasets.
- **Hunyuan-GameCraft-2 Model:** A 14B parameter `MoE` model that integrates text prompts, keyboard, and mouse signals into a single controllable framework.
- **Real-time Performance:** Through engineering optimizations (quantization, parallelization), the model generates video at 16 FPS, suitable for interactive use.
- **InterBench:** A comprehensive benchmark for evaluating interaction quality beyond simple visual fidelity.
- **Key Finding:** The model demonstrates superior generalization, handling unseen interactions (like "drawing a phone" or "summoning a dragon") despite these not being explicitly in the training set.

  ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, a novice should be familiar with:
- **Diffusion Models:** A class of generative models that learn to create data by reversing a "noising" process. They start with random noise and iteratively refine it into a clear image or video.
- **Mixture-of-Experts (MoE):** Instead of using one giant neural network for every task, an `MoE` model has multiple "experts" (smaller sub-networks). A "router" decides which expert is best suited for a specific part of the input, making the model more efficient and powerful.
- **Autoregressive Generation:** A process where the model generates data one piece at a time (e.g., one video frame after another), using its previous outputs as context for the next step.
- **KV Cache (Key-Value Cache):** A technique used in `Transformers` to store mathematical results from previous steps so the model doesn't have to recompute them. This is essential for speed during long-sequence generation.
- **Flow Matching:** A modern training objective for diffusion models that learns the "velocity" or "direction" needed to transform noise into data, often leading to faster and more stable training than traditional methods.

## 3.2. Previous Works
The paper builds on several lineages:
- **Genie & GameGen:** Early "world models" that used latent actions or discrete keyboard signals to simulate game physics.
- **HunyuanVideo:** The base 14B model upon which this work is built.
- **Self-Forcing:** A training technique where a model is trained on its own previous (potentially erroneous) predictions to learn how to correct itself, bridging the gap between training and real-world usage.

## 3.3. Technological Evolution
Early game AI focused on 3D engines (Unreal/Unity) where rules were hard-coded. Generative AI shifted this to "video-based world models," where the AI *imagines* the pixels of the game. `Hunyuan-GameCraft-2` pushes this further by adding "instruction following," moving from simple physics simulation to complex semantic understanding.

## 3.4. Differentiation Analysis
Unlike previous models like `GameNGen` (which only used keyboard inputs) or `Genie` (which focused on 2D physics), `Hunyuan-GameCraft-2` allows **open-ended text instructions**. It doesn't just move the camera; it changes the state of the world (e.g., making it snow or causing an explosion) based on a sentence.

---

# 4. Methodology

## 4.1. Principles
The core idea is to treat "interaction" as a conditioned generation task. The model is given a starting frame and an instruction; it then predicts the subsequent frames that satisfy the instruction while maintaining physical and temporal consistency.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Defining and Constructing Interactive Data
The authors argue that "interactive" data must show a clear "before" and "after" state. They define an `Interaction Caption` $I_{t, t+1}$ as the semantic difference between two sequential video clips.
The relationship is expressed as:
\$
I_{t, t+1} = \Delta ( \Phi ( C_{t+1} ) , \Phi ( C_t ) )
\$
- $C_t$ and $C_{t+1}$: The standard captions describing the visual content of clip $t$ and clip $t+1$.
- $\Phi$: A semantic encoder (like a large language model) that turns text into a mathematical representation.
- $\Delta$: A difference operator that identifies what changed (e.g., if $C_t$ is "a closed door" and $C_{t+1}$ is "an open door," $I_{t, t+1}$ becomes "opening the door").

  The following figure (Figure 4 from the original paper) illustrates how these captions are structured:

  ![该图像是示意图，展示了如何生成标准描述和交互描述。左侧步骤展示了处理后的视频片段如何转化为标准描述，而右侧则展示了两段描述之间的差异（Diff）如何生成交互描述。](images/4.jpg)
  *该图像是示意图，展示了如何生成标准描述和交互描述。左侧步骤展示了处理后的视频片段如何转化为标准描述，而右侧则展示了两段描述之间的差异（Diff）如何生成交互描述。*

### 4.2.2. Model Architecture: The Action-Injected MoE
The model is built on a 14B `image-to-video MoE` foundation. It takes three types of inputs:
1.  **Initial Frame:** To set the scene.
2.  **Keyboard/Mouse Signals:** Converted into `Plücker embeddings` (a mathematical way to represent 3D lines/camera rays) to control camera movement.
3.  **Interaction Instructions:** Natural language processed by a `Multimodal Large Language Model (MLLM)` to guide specific actions.

    The following figure (Figure 5 from the original paper) shows the overall architecture:

    ![该图像是架构图，展示了Hunyuan-GameCraft-2模型的工作流程。图中包含指令处理、图像条件、噪声、键盘动作等模块，通过自注意力和交叉注意力机制实现指令驱动的互动任务。](images/5.jpg)
    *该图像是架构图，展示了Hunyuan-GameCraft-2模型的工作流程。图中包含指令处理、图像条件、噪声、键盘动作等模块，通过自注意力和交叉注意力机制实现指令驱动的互动任务。*

### 4.2.3. Autoregressive Distillation and Long-Video Tuning
To make the model fast and capable of long videos, the authors use a `Distillation` process. They take a slow, high-quality model (the teacher) and train a faster version (the student) to mimic it.
A critical part of this is `Randomized Extended Long-Video Tuning`. The model performs a "rollout" where it generates many frames, and the training objective ensures that these frames don't "drift" or become blurry over time.
The training uses a `DMD (Distributional Moment Distance)` loss:
\$
\mathcal{L} = \mathrm{DMD} \left( T_{\mathrm{fake}} ( x_t ( W ) , t , c_{\mathrm{student}} ) , T_{\mathrm{real}} ( x_t ( W ) , t , c_{\mathrm{teacher}} ) \right)
\$
- $W$: A window of generated frames.
- $T_{\mathrm{fake}}$ and $T_{\mathrm{real}}$: "Teacher" models that score how realistic the generated content is compared to real data.
- $c_{\mathrm{student}}$ and $c_{\mathrm{teacher}}$: The conditions (history) used by the student and teacher.

  This logic is formalized in `Algorithm 1` from the paper:

**Algorithm 1: Randomized Extended Long-Video Tuning**
1. Sample a ground truth video $V_{gt}$.
2. Randomize a rollout length $N$.
3. **Autoregressive Rollout:** Generate frames chunk by chunk using the `KV Cache`.
4. **Randomized Window Sampling:** Pick a random segment from the generated video.
5. **Interleaved Forcing:** Compare the student's prediction (based on its own history) against the teacher's prediction (based on ground truth).
6. **Update:** Minimize the `DMD` loss to align the student with the teacher.

### 4.2.4. Multi-turn Interaction and KV-recache
For a smooth user experience, the model uses a `ReCache` mechanism. When a user gives a *new* instruction mid-video, the model doesn't start from scratch. It recomputes only the most recent "block" of mathematical data in its memory (the `KV Cache`) to incorporate the new instruction, allowing for immediate and accurate response.

---

# 5. Experimental Setup

## 5.1. Datasets
- **Gameplay Data:** Collected from over 150 AAA games (e.g., *Cyberpunk 2077*, *Assassin's Creed*). This provides diversity in lighting and style.
- **Synthetic Data:** Since real data for "opening doors" or "explosions" is hard to get, they used a `VLM` to guide an image-editing model to create "start" and "end" frames, then used a video model to fill in the transition.
- **Scale:** 1M game-play clips for initial training, 150K high-quality interactive samples for fine-tuning.

## 5.2. Evaluation Metrics
The authors use standard metrics and their new `InterBench` suite.

1.  **FVD (Fréchet Video Distance):** Measures how similar the distribution of generated videos is to real videos.
    \$
    \mathrm{FVD} = | \mu_r - \mu_g |^2 + \mathrm{Tr} ( \Sigma_r + \Sigma_g - 2 ( \Sigma_r \Sigma_g )^{1/2} )
    \$
    - $\mu_r, \Sigma_r$: Mean and covariance of features from real videos.
    - $\mu_g, \Sigma_g$: Mean and covariance of features from generated videos.

2.  **RPE (Relative Pose Error):** Quantifies how accurately the camera follows the requested path.
    - `RPE_trans`: Error in translation (moving).
    - `RPE_rot`: Error in rotation (turning).

3.  **InterBench Dimensions:**
    - **Trigger Rate:** Did the action actually happen? (Binary: 0 or 1).
    - **Prompt-Video Alignment:** How well does the video match the text description? (Score 0-5).
    - **Interaction Fluency:** Is the motion smooth without "teleporting" objects? (Score 0-5).
    - **Interaction Scope Accuracy:** Do global actions (rain) affect the whole scene and local actions (torch) affect only the nearby area? (Score 0-5).
    - **End-State Consistency:** Does the result of the action (e.g., an open door) stay that way? (Score 0-5).
    - **Object Physics Correctness:** Do objects maintain their shape and follow gravity? (Score 0-5).

## 5.3. Baselines
The model is compared against:
- **HunyuanVideo:** The generic video foundation model.
- **Wan2.2 A14B:** A leading open-source video model.
- **LongCatVideo:** A model specialized in long-sequence consistency.
- **Matrix-Game:** A previous state-of-the-art interactive model.

  ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
`Hunyuan-GameCraft-2` significantly outperforms baselines in interaction accuracy. While models like `Wan2.2` are good at general video, they often ignore specific commands like "open the door" or create "ghosting" effects. `GameCraft-2` achieves a **Trigger Rate** of nearly 98% for actor actions.

The following are the results from Table 1 of the original paper, comparing various world models:

<table>
<thead>
<tr>
<th>Model</th>
<th>Resolution</th>
<th>Training Data</th>
<th>Action type</th>
<th>Action space</th>
<th>Generalizable</th>
<th>Real time</th>
</tr>
</thead>
<tbody>
<tr>
<td>GameNGen</td>
<td>240p</td>
<td>Gameplay</td>
<td>Keyboard</td>
<td>Closed</td>
<td>X</td>
<td>V</td>
</tr>
<tr>
<td>Oasis</td>
<td>360p</td>
<td>Gameplay video</td>
<td>Key+Mouse</td>
<td>Closed</td>
<td>X</td>
<td>V</td>
</tr>
<tr>
<td>GameCraft-1</td>
<td>720p</td>
<td>Gameplay + Rendered</td>
<td>Key+Mouse</td>
<td>Closed</td>
<td>V</td>
<td>X</td>
</tr>
<tr>
<td><strong>GameCraft-2</strong></td>
<td><strong>480p</strong></td>
<td><strong>Gameplay + Synthetic</strong></td>
<td><strong>Key+Mouse+Prompt</strong></td>
<td><strong>Open-ended</strong></td>
<td><strong>V</strong></td>
<td><strong>V (16 FPS)</strong></td>
</tr>
</tbody>
</table>

## 6.2. Interaction Performance (InterBench)
The model's primary strength is the "fidelity" of the interaction—meaning the interaction looks physically "correct."

The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Category</th>
<th rowspan="2">Method</th>
<th colspan="6">InterBench Dimensions</th>
</tr>
<tr>
<th>Trigger↑</th>
<th>Align↑</th>
<th>Fluency↑</th>
<th>Scope↑</th>
<th>EndState↑</th>
<th>Physics↑</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">Actor Actions</td>
<td>Wan2.2 A14B</td>
<td>0.836</td>
<td>3.490</td>
<td>3.488</td>
<td>4.036</td>
<td>4.054</td>
<td>3.175</td>
</tr>
<tr>
<td><strong>GameCraft-2</strong></td>
<td><strong>0.983</strong></td>
<td><strong>4.087</strong></td>
<td><strong>4.191</strong></td>
<td><strong>4.576</strong></td>
<td><strong>4.686</strong></td>
<td><strong>3.828</strong></td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies
The authors found that `Long-Video Tuning` is essential. Without it, the video quality degrades significantly after about 450 frames. Increasing the `Sink Token` size (keeping the first frame in memory) also helps maintain the 3D coordinate system, preventing the camera from "getting lost."

As seen in the following figure (Figure 16 from the paper), tuning prevents the background from "melting" during long sequences:

![Figure 16. Qualitative Analysis of Long-Video Tuning and Cache Settings. Row 1: Baseline results without Long-Video Tuning (sink token size $= 1$ , local attention $\\mathrm { s i z e } = 6$ ). Row 2: Incorporates Long-Video Tuning upon the baseline. Row 3: Further modifies setting based on Row 2 by increasing the sink token size to 3 and local attention size to 9. Input prompts and camera parameters remain consistent across all samples.](images/16.jpg)
*该图像是插图，展示了长视频调优和缓存设置的定性分析。图中分为三行，第一行为未调优的基准结果，第二行为在基准上应用长视频调优的结果（sink token大小为1，local attention大小为6），第三行为在第二行的基础上进一步调整，sink token大小为3及local attention大小为9。每帧（Frame 50、Frame 150、Frame 250、Frame 450）的画面展示了不同时间点的场景变化。*

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
`Hunyuan-GameCraft-2` represents a shift from "video generation" to "interactive simulation." By defining `Interactive Video Data` and using an `MoE` architecture with `autoregressive distillation`, it creates a world that is both visually high-fidelity and semantically responsive. It achieves real-time speeds (16 FPS) and demonstrates a remarkable ability to generalize to new, unseen instructions.

## 7.2. Limitations & Future Work
- **Semantic Drift:** Even with tuning, very long videos (over 500 frames) can still drift away from the original intent.
- **Memory Capacity:** The model lacks a "global memory" bank; it only remembers what's in its current `KV Cache`.
- **Reasoning:** It handles immediate actions well but cannot yet handle "multi-stage" logical tasks (e.g., "Find the key, then open the door, then get the treasure").
- **Hardware:** 16 FPS is good, but for "twitch" gaming (high reactivity), even lower latency and higher frame rates are needed.

## 7.3. Personal Insights & Critique
This paper is highly significant because it solves the "What do we train on?" problem for interactive AI. By creating a synthetic data pipeline, the authors bypass the need for expensive manual gameplay recording. 

However, a potential critique is the **resolution vs. speed trade-off**. To reach 16 FPS, the model operates at 480p. While acceptable for a prototype, modern games are 1080p or 4K. The jump in computational cost to reach those resolutions while maintaining 16+ FPS remains a massive hurdle. Furthermore, the reliance on a "Sink Token" (the first frame) might limit the model's ability to simulate *drastic* world changes where the original starting point is no longer visible or relevant. Overall, this is a robust step toward the "holodeck" vision of AI-generated entertainment.