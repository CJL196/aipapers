# 1. Bibliographic Information

## 1.1. Title
Advancing Open-source World Models

## 1.2. Authors
The paper is authored by the Robbyant Team. Key contributors include:
*   **Base Model:** Zelin Gao, Qiuyu Wang, Yinghao Xu, Shuailei Ma.
*   **Post Training:** Yanhong Zeng, Jiapeng Zhu.
*   **Data Acquisition & Pipeline:** Ka Leong Cheng, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Yixuan Li, Jiayi Zhu, Hanlin Wang, Yihao Meng, Kecheng Zheng.
*   **Applications:** Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu.
*   **Leadership:** Xing Zhu (Sponsor), Yujun Shen (Lead), Hao Ouyang (Lead).
    The team appears to be a specialized research group focused on large-scale generative models and computer vision.

## 1.3. Journal/Conference
This paper was published as a technical report on **arXiv** (2601.20540). While arXiv is a preprint server and not a peer-reviewed journal, it is the standard venue for rapid dissemination of high-impact AI research from top-tier labs.

## 1.4. Publication Year
The paper was published on **January 28, 2026**.

## 1.5. Abstract
The authors present **LingBot-World**, an open-source world simulator derived from video generation technology. The model is designed to bridge the gap between open-source and proprietary (closed-source) "world models"—AI systems that can simulate the physical and logical rules of an environment. LingBot-World features high-fidelity visual synthesis across diverse styles (realism, sci-fi, cartoon), maintains "long-term memory" (consistency over minute-long horizons), and supports real-time interactivity with sub-second latency at 16 frames per second (fps). The release includes code, model weights, and datasets to empower the community in fields like gaming, robotics, and content creation.

## 1.6. Original Source Link
*   **Original Source Link:** [https://arxiv.org/abs/2601.20540](https://arxiv.org/abs/2601.20540)
*   **PDF Link:** [https://arxiv.org/pdf/2601.20540v1](https://arxiv.org/pdf/2601.20540v1)
*   **Publication Status:** Preprint (Technical Report).

# 2. Executive Summary

## 2.1. Background & Motivation
The "holy grail" of computer vision is developing an AI that understands the physical world. While recent **video generation models** (like Sora or Kling) can create visually stunning clips, they often lack **interactivity** and **logical consistency** over long periods. Transitioning from a video generator to a **world simulator** (a model an agent can actually "live" in and interact with) faces three major hurdles:
1.  **Data Bottleneck:** It is difficult to find large datasets that pair video frames with the specific actions (like "turn left" or "open door") that caused the scene to change.
2.  **Long-term Consistency:** Models often "forget" what the room looked like after a few seconds of movement, leading to "hallucinations" where objects disappear or change shape.
3.  **Inference Latency:** Generating high-quality video is computationally slow, making real-time interaction (like playing a video game) impossible with standard architectures.

    LingBot-World aims to solve these issues by providing a scalable data engine and a multi-stage training pipeline that converts a standard video generator into a real-time, action-controllable world model.

## 2.2. Main Contributions / Findings
*   **Scalable Data Engine:** A framework that combines real-world videos, game engine data (Unreal Engine), and a **hierarchical captioning strategy** to disentangle motion from scene content.
*   **Three-Stage Training Pipeline:** A progressive approach moving from **Pre-training** (general video knowledge) to **Middle-training** (action control and long-term memory) and finally **Post-training** (efficiency and real-time inference).
*   **28B Parameter Mixture-of-Experts (MoE) Architecture:** A massive model that uses specialized "expert" sub-networks to handle different parts of the generation process, maintaining high quality without skyrocketing the computational cost.
*   **Real-time Performance:** Achieving sub-second latency and 16 fps, making it one of the first high-fidelity world models capable of live user interaction.
*   **Open-source Commitment:** Releasing code, weights, and the data pipeline to democratize access to advanced world modeling technology.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. World Models
A **world model** is an AI system that internalizes the "rules" of an environment. Unlike a simple image generator, a world model predicts the next state of the world given the current state and a specific action. For example, if the user "presses the forward key," the world model predicts the visual update of moving closer to an object.

### 3.1.2. Diffusion Models
The foundational technology here is the **Diffusion Model**. Imagine starting with a clear image and slowly adding static (noise) until it is unrecognizable. A diffusion model learns to reverse this process, starting from pure noise and "denoising" it step-by-step to create a coherent image or video.

### 3.1.3. Mixture-of-Experts (MoE)
`MoE` is a neural network architecture where, instead of one giant network doing all the work, the model is split into many smaller "experts." A "router" decides which expert is best suited for a specific task or data point. This allows the model to have a huge number of total parameters (28B in this case) while only activating a fraction of them during inference, saving energy and time.

### 3.1.4. Autoregressive Generation
In video, **autoregressive** means the model generates the next frame based on the sequence of all previous frames. This is crucial for consistency, as the model "remembers" its own past outputs to ensure the future makes sense.

## 3.2. Previous Works
The paper builds on several key milestones:
*   **Sora / Wan2.1:** Large-scale video generation models that use **Spatio-Temporal Transformers**. Transformers use an **Attention mechanism** to decide which parts of a video sequence are most relevant. The standard formula for Attention is:
    \$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    \$
    Where $Q$ (Query), $K$ (Key), and $V$ (Value) are representations of the input data, and $d_k$ is a scaling factor.
*   **Genie / Matrix-Game:** Early attempts at interactive world models. `Genie` focused on platformer-style games, while `Matrix-Game` explored 3D environments but often struggled with high-resolution realism or long-term consistency.

## 3.3. Technological Evolution
The field has evolved from **GANs** (Generative Adversarial Networks) which were fast but unstable, to **Diffusion Models** which are high-quality but slow, and now toward **Interactive World Models**. This paper represents the shift toward combining high-quality diffusion with the speed and causality of autoregressive systems.

## 3.4. Differentiation Analysis
Compared to models like `Mirage 2` or `Genie 3`, LingBot-World distinguishes itself by:
1.  **Openness:** Most high-tier world models (like those from Google or OpenAI) are closed-source.
2.  **Horizon:** It supports "minute-level" consistent generation, whereas many others drift or fail after 10-20 seconds.
3.  **Control:** It uses a hybrid action representation (discrete and continuous) for more precise movement control.

# 4. Methodology

## 4.1. Principles
The core intuition of LingBot-World is that a world model should be trained like a language model but for "visual tokens." By treating video frames as a sequence of states influenced by actions, the model learns a **transition function** of the universe it is simulating.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Problem Formulation
The model aims to learn the transition dynamics of a world. We define a sequence of video frames as $\mathcal{V} = \{x_1, x_2, ..., x_T\}$ and a sequence of user actions as $\mathcal{A} = \{a_1, a_2, \ldots, a_T\}$. The model $\theta$ is optimized to maximize the likelihood of future states:

$$
\operatorname*{max}_{\theta} \mathbb{E} \left[ \log p_{\theta} (x_{t:t+L} \mid x_{<t}, a_{t:t+L}) \right]
$$

In this formula:
*   $x_{t:t+L}$ represents the frames to be predicted in the future (from time $t$ to $t+L$).
*   $x_{<t}$ represents the history of frames seen so far.
*   $a_{t:t+L}$ represents the sequence of actions the user takes during those future frames.
*   $L \geq 1$ is the **prediction horizon** (how far into the future the model looks).

### 4.2.2. Stage I: Pre-Training (The Foundation)
The model begins with a 14B-parameter version of `Wan2.1`, a powerful video generator. This stage ensures the model understands **general video priors**—how water ripples, how light reflects, and how objects move naturally. This provides the "visual IQ" necessary before adding interactive logic.

### 4.2.3. Stage II: Middle-Training (World Knowledge Injection)
In this stage, the model is upgraded to a **Mixture-of-Experts (MoE)** architecture with 28B parameters. The authors use a **Progressive Curriculum Training** approach, starting with short 5-second videos and gradually increasing to 60-second sequences.

To handle actions, they introduce an **Adaptive Layer Normalization (AdaLN)** mechanism. This allows the model to inject action signals $a$ directly into the neural network's layers, adjusting the "style" of the generation (i.e., the direction of movement) without destroying the pre-trained visual quality.

The following figure (Figure 4 from the original paper) illustrates this multi-stage training flow:

![该图像是示意图，展示了LingBot-World的训练阶段。图中分为三个阶段：第一阶段为预训练，目标为建立一般视频先验，强调开放域生成和高保真纹理；第二阶段为中间训练，重点为世界知识注入，包括交互逻辑和长期动态；第三阶段为后训练，强调实时交互，旨在实现低延迟和因果性。每个阶段的关键特点均有详细列出。](images/4.jpg)
*该图像是示意图，展示了LingBot-World的训练阶段。图中分为三个阶段：第一阶段为预训练，目标为建立一般视频先验，强调开放域生成和高保真纹理；第二阶段为中间训练，重点为世界知识注入，包括交互逻辑和长期动态；第三阶段为后训练，强调实时交互，旨在实现低延迟和因果性。每个阶段的关键特点均有详细列出。*

### 4.2.4. Stage III: Post-Training (Real-Time Optimization)
To make the model interactive, the authors switch from **bidirectional attention** (which looks at the whole video at once) to **causal attention** (which only looks at the past). This allows for **KV Caching** (Key-Value Caching), a technique where the model stores the computations of previous frames so it doesn't have to re-calculate them for every new frame.

To maintain quality with fewer steps, they use **Distribution Matching Distillation (DMD)**. The goal is to make a "student" model (fast) behave exactly like the "teacher" model (slow but high-quality). The gradient used to update the student $\theta$ is:

$$
\nabla_{\theta} \mathbb{E}_{t} \big[ D_{\mathrm{KL}} \big( p_{\theta, t} \| p_{\mathrm{data}, t} \big) \big] = - \mathbb{E}_{t, \hat{x}_{t} \sim q_{t \mid 0} (\hat{x}_{t} \mid \bar{x}), \bar{x} \sim p_{\theta} (\bar{x} \mid a)} \left[ \big ( s_{\mathrm{real}} \big ( \hat{x}_{t}, t, a \big ) - s_{\mathrm{fake}} \big ( \hat{x}_{t}, t, a \big ) \big ) \frac{\partial \hat{x}}{\partial \theta} \right]
$$

Where:
*   $p_{\theta, t}$ is the student's output distribution.
*   $p_{\mathrm{data}, t}$ is the target data distribution.
*   $s_{\mathrm{real}}$ and $s_{\mathrm{fake}}$ are the "score functions" (gradients) of the teacher and student, respectively. This formula essentially pulls the student's output toward the high-quality manifold of the teacher.

    The student model is also trained with an **Adversarial Loss** to ensure visual sharpness. The discriminator $D$ and generator $G$ losses are:

$$
\begin{array}{rl} & \mathcal{L}_G = \mathbb{E}_{p(\tilde{x})} [f(1 - D(\mu_{\mathrm{fake}}(\tilde{x}_t, t, a)))] , \\ & \mathcal{L}_D = \mathbb{E}_{p(x)} [f(D(\mu_{\mathrm{fake}}(x_t, t, a)))] - \mathbb{E}_{p(\tilde{x})} [f(1 - D(\mu_{\mathrm{fake}}(\tilde{x}_t, t, a)))] , \end{array}
$$

Where $f(\cdot)$ is the `softplus` function, $x$ are real videos, and $\tilde{x}$ are synthesized videos. This "push-and-pull" training ensures the model stays realistic even during long rollouts.

The following diagram (Figure 6 from the original paper) shows the architecture of this distilled real-time system:

![该图像是示意图，展示了DiTBlock生成器和鉴别器架构的结构。生成器使用块因果注意力，并初始化自高噪声专家。图中展示了文本嵌入、图像/视频条件、动作和噪声潜变量的功能。该图也包含了鉴别器部分，利用交叉注意力进行处理。](images/6.jpg)
*该图像是示意图，展示了DiTBlock生成器和鉴别器架构的结构。生成器使用块因果注意力，并初始化自高噪声专家。图中展示了文本嵌入、图像/视频条件、动作和噪声潜变量的功能。该图也包含了鉴别器部分，利用交叉注意力进行处理。*

# 5. Experimental Setup

## 5.1. Datasets
The authors built a massive, multi-source dataset:
*   **General Video Curator:** Millions of real-world videos (human/animal ego-centric and third-person perspectives).
*   **Game Data:** High-fidelity data from **Unreal Engine**, where RGB frames are strictly paired with camera positions (intrinsics/extrinsics).
*   **Synthetic Rendering:** A pipeline that creates randomized but physically plausible camera trajectories (loops, 360-degree turns, etc.) to teach the model spatial memory.

### 5.1.1. Data Example: Hierarchical Captioning
To help the model learn, they don't just use one caption. For a single video, they generate:
1.  **Narrative Caption:** A story-like description (e.g., "The camera pans right toward a majestic white statue...").
2.  **Scene-Static Caption:** Describes only objects (e.g., "A courtyard with East Asian architecture, red doors, and stone pavement").
3.  **Dense Temporal Caption:** A log of events with timestamps.

    The following figure (Figure 3 from the original paper) depicts this data profiling and captioning pipeline:

    ![该图像是一个示意图，展示了从原始数据到最终数据的处理过程。流程包括基本过滤与切片、语义分析以及层级字幕生成，每个步骤将数据逐步转化为可用的信息和说明。](images/3.jpg)
    *该图像是一个示意图，展示了从原始数据到最终数据的处理过程。流程包括基本过滤与切片、语义分析以及层级字幕生成，每个步骤将数据逐步转化为可用的信息和说明。*

## 5.2. Evaluation Metrics
The paper uses the **VBench** suite, a comprehensive benchmark for video generation. Key metrics include:
1.  **Imaging Quality:**
    *   **Definition:** Measures the clarity, sharpness, and lack of visual artifacts in the generated frames.
2.  **Aesthetic Quality:**
    *   **Definition:** Evaluates the artistic appeal, lighting, and composition of the video based on human preference models.
3.  **Dynamic Degree:**
    *   **Definition:** Quantifies how much meaningful movement and action occur in the video (to prevent "static" videos).
4.  **Temporal Flickering:**
    *   **Definition:** Measures how much the video "shakes" or changes inconsistently between frames. Higher scores usually mean smoother video.
5.  **Overall Consistency:**
    *   **Definition:** A holistic score of how well the video maintains its subject and environment over time.

## 5.3. Baselines
The model is compared against:
*   **Matrix-Game 2.0:** A recent interactive world model focused on gaming.
*   **Yume-1.5:** A text-controlled interactive world generation model.
*   **HY-World 1.5:** Another general-domain interactive world model.
*   **Genie 3:** Google's latest frontier world model (closed-source).

# 6. Results & Analysis

## 6.1. Core Results Analysis
LingBot-World outperforms existing open-source models across almost all categories, particularly in **Dynamic Degree** and **Overall Consistency**. It achieves a Dynamic Degree of 0.8857, significantly higher than Yume-1.5 (0.7612), indicating that LingBot-World creates much more "alive" and reactive environments.

The following are the results from Table 1 of the original paper, comparing the features of various models:

<table>
<thead>
<tr>
<th>Model</th>
<th>Domain</th>
<th>Generation Horizon</th>
<th>Dynamic Degree</th>
<th>Resolution</th>
<th>Real-time</th>
<th>Open-source</th>
</tr>
</thead>
<tbody>
<tr>
<td>Matrix-Game 2.0</td>
<td>Game</td>
<td>Short</td>
<td>Low</td>
<td>480p</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Yume-1.5</td>
<td>General</td>
<td>Short</td>
<td>Low</td>
<td>480p</td>
<td>X</td>
<td>√</td>
</tr>
<tr>
<td>HY-World 1.5</td>
<td>General</td>
<td>Medium</td>
<td>Low</td>
<td>720p</td>
<td>√</td>
<td>√</td>
</tr>
<tr>
<td>Mirage 2</td>
<td>General</td>
<td>Long</td>
<td>Medium</td>
<td>480p</td>
<td>√</td>
<td>X</td>
</tr>
<tr>
<td>Genie 3</td>
<td>General</td>
<td>Long</td>
<td>Medium</td>
<td>720p</td>
<td>√</td>
<td>X</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td>General</td>
<td>Long</td>
<td>High</td>
<td>720p</td>
<td>✓</td>
<td>✓</td>
</tr>
</tbody>
</table>

The following are the quantitative VBench results from Table 2 of the original paper:

| Model | Imaging Quality | Aesthetic Quality | Dynamic Degree | Motion Smooth | Temporal Flickering | Overall Consistency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Yume-1.5 | 0.5838 | 0.5185 | 0.7612 | 0.9709 | 0.9545 | 0.1994 |
| HY-World 1.5 | 0.6512 | 0.5487 | 0.7217 | 0.9897 | 0.9773 | 0.2016 |
| **Ours** | **0.6683** | **0.5660** | **0.8857** | 0.9895 | 0.9648 | **0.2178** |

## 6.2. Qualitative Analysis
The model's ability to maintain consistency is demonstrated in Figure 12. Even when objects (like a car) move out of the camera's view and then return, the model "remembers" their existence and position, a hallmark of a true world model.

As seen in Figure 10 and 11, the "Fast" version of the model maintains high fidelity even at 16 fps:

![Figure 10. Qualitative results of LingBot-World-Fast .](images/10.jpg)
*该图像是图表，展示了LingBot-World的定性结果，包括多种场景的视觉效果，如自然风景、城市环境及水面反射等。这些图像体现了系统在高保真度和动态效果上的强大能力，适用于多种应用场景。*

![Figure 11. Qualitative results of LingBot-World-Fast .](images/11.jpg)
*该图像是图表，展示了LingBot-World-Fast的定性结果。图中包含多个场景，展现了高保真度和丰富的动态效果，涵盖了现实、科学和卡通等多种风格。*

## 6.3. Emergent Memory Capability
One of the most impressive findings is the emergent **spatial memory**. The model can generate a consistent 3D room for up to 10 minutes. By feeding the generated video into a 3D reconstruction tool (like Gaussian Splatting), the authors show that the model's "mental map" of the room is geometrically accurate, as seen in Figure 16:

![该图像是一个示意图，展示了多种环境下的3D场景生成，包括室内及城市景观。它体现了高保真度和动态表现，展示了不同视觉效果的转变，为模拟和交互提供了视角。](images/16.jpg)
*该图像是一个示意图，展示了多种环境下的3D场景生成，包括室内及城市景观。它体现了高保真度和动态表现，展示了不同视觉效果的转变，为模拟和交互提供了视角。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
LingBot-World establishes a new high-water mark for open-source world models. By combining a massive 28B MoE architecture with a clever three-stage training pipeline and a sophisticated data engine, the authors have created a system that is not only visually stunning but also interactive and logically consistent over long periods. It successfully bridges the gap between static video generation and dynamic world simulation.

## 7.2. Limitations & Future Work
Despite its successes, the authors identify several challenges:
*   **Memory Stability:** The model's memory is "implicit" (stored in the neural weights) rather than an explicit database. This can still lead to "drifting" or forgetting over very long sessions (e.g., hours).
*   **Computational Cost:** Running a 28B parameter model requires enterprise-grade GPUs (like NVIDIA H100s), limiting its use on consumer hardware.
*   **Action Space:** Current controls are limited to navigation (W, A, S, D) and looking around. Complex interactions like "picking up an object and throwing it" are not yet fully supported.
*   **Precision:** Fine-grained grounding (e.g., clicking on a tiny button) remains difficult for the current architecture.

## 7.3. Personal Insights & Critique
The most significant contribution of this paper is the **disentanglement of motion from scene content** via hierarchical captioning. This is a brilliant solution to the "mushy" dynamics often seen in video models. By explicitly telling the model what is "scene" and what is "camera movement" during training, it learns a much cleaner internal representation of space.

However, a potential issue is the reliance on **distillation**. While DMD makes the model fast, distilled models can sometimes lose the "creativity" or "diversity" of the original teacher model. It remains to be seen how well this model generalizes to extremely weird or "out-of-distribution" prompts compared to the base Wan2.1 model. Overall, LingBot-World is a massive win for the open-source community, providing a robust foundation for future work in AI-driven gaming and robotics.