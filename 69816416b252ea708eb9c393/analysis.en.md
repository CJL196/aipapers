# 1. Bibliographic Information

## 1.1. Title
DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation

## 1.2. Authors
Haozhe Xie\*, Beichen Wen\*, Jiarui Zheng, Zhaoxi Chen, Fangzhong Hong, Haiwen Diao, Ziwei Liu. The authors are affiliated with S-Lab, Nanyang Technological University. (\* indicates equal contribution).

## 1.3. Journal/Conference
The paper is published on arXiv, a preprint server, and is scheduled to be presented at an unspecified venue. While arXiv itself is not a peer-reviewed journal or conference, it is a widely used platform for disseminating early research findings in computer science, physics, mathematics, and other fields. Papers on arXiv are often later submitted to and published in prestigious conferences (like NeurIPS, ICML, CVPR, RSS, CoRL) or journals, which are highly reputable venues in robotics, machine learning, and computer vision. The `Published at (UTC): 2026-01-29T18:59:51.000Z` suggests it is a very recent or upcoming work.

## 1.4. Publication Year
2026 (as indicated by the publication timestamp `2026-01-29T18:59:51.000Z` and references like `arXiv, 2601.22153, 2026`).

## 1.5. Abstract
The paper introduces `DynamicVLA`, a novel framework designed to address the challenges of manipulating dynamic objects using `Vision-Language-Action (VLA)` models. While `VLA` models excel in static manipulation, they struggle with dynamic scenarios due to requirements for rapid perception, temporal anticipation, and continuous control. `DynamicVLA` integrates temporal reasoning and closed-loop adaptation through three core designs: 1) a compact `0.4B` parameter `VLA` model utilizing a convolutional vision encoder for efficient, structurally faithful encoding and fast multimodal inference; 2) `Continuous Inference`, a pipelined execution scheme that overlaps reasoning and action execution to reduce latency; and 3) `Latent-aware Action Streaming`, which bridges the perception-execution gap by ensuring temporally aligned action execution. To facilitate research in this underexplored area, the authors also introduce the `Dynamic Object Manipulation (DOM)` benchmark, created with an auto data collection pipeline that efficiently gathers `200K` synthetic episodes and `2K` real-world episodes without teleoperation. Extensive evaluations demonstrate `DynamicVLA`'s significant improvements in response speed, perception, and generalization, positioning it as a unified framework for dynamic object manipulation across various robot embodiments.

## 1.6. Original Source Link
Official Source: https://arxiv.org/abs/2601.22153
PDF Link: https://arxiv.org/pdf/2601.22153v1
Publication Status: Preprint on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The paper addresses a significant challenge in robotics: `dynamic object manipulation`. This refers to tasks where a robot must interact with objects that are in continuous motion, such as handing items, repositioning moving objects, or stabilizing them. This problem is inherently more complex and demanding than `static manipulation` (where objects are stationary during interaction), as it requires the robot to:
*   **Rapidly perceive:** Understand the changing state of the environment in real-time.
*   **Temporally anticipate:** Predict the future motion of objects to plan actions effectively.
*   **Continuously control:** Execute precise and adaptive actions without interruption.

    Existing `Vision-Language-Action (VLA)` models, despite their strong generalization capabilities in `static manipulation` tasks, largely fail in dynamic scenarios. This failure is primarily attributed to `inference latency`, the delay between receiving sensory input, processing it through the model, and generating an action. In dynamic environments, even minor latency can cause a mismatch between the robot's perception of the object's state and the actual state of the object when the action is executed, leading to task failure. Prior `VLA` models, often built on large `3B-7B` parameter `Vision-Language Models (VLMs)`, were not designed with the stringent real-time requirements of dynamic manipulation in mind. While some recent works touch on moving targets, they often rely on highly structured settings or tasks tolerant to spatial and timing errors, not the `precise 6DoF control` (six degrees of freedom, referring to position along x, y, z axes and rotation around x, y, z axes) needed for general dynamic manipulation.

The paper's entry point is to bridge this critical `perception-execution gap` in dynamic object manipulation by explicitly addressing `inference latency` and `temporal misalignment`.

## 2.2. Main Contributions / Findings
The paper makes several significant contributions to the field of robotics and `VLA` models:

*   **A Compact 0.4B VLA Model for Fast Inference:** `DynamicVLA` introduces a highly efficient `VLA` model with only `0.4 billion` parameters. It uses a `convolutional vision encoder` (`FastViT`) instead of more common transformer-based ones, which offers `spatially efficient` and `structurally faithful encoding`. This compact design is crucial for achieving `fast multimodal inference`, which is a prerequisite for real-time responsiveness in dynamic environments.
*   **Novel Closed-Loop Adaptation Mechanisms:**
    *   **Continuous Inference (CI):** This design introduces a `pipelined execution scheme` where the `VLA` model's reasoning (inference) and the robot's action execution `overlap`. This eliminates `inter-chunk waiting` (idle time between action sequences), allowing for a continuous stream of actions and timely adaptation to object motion.
    *   **Latent-aware Action Streaming (LAAS):** This mechanism explicitly addresses the `perception-execution gap` caused by inference delay. It ensures `temporally aligned action execution` by discarding outdated predicted actions and prioritizing the most recent predictions available at each timestep. This allows the robot to adapt promptly to the evolving environment.
*   **Dynamic Object Manipulation (DOM) Benchmark:** To overcome the critical lack of large-scale datasets for dynamic manipulation, the paper introduces `DOM`. This benchmark is built from scratch with an `auto data collection pipeline` capable of efficiently gathering:
    *   `200K synthetic episodes` across `2.8K scenes` and `206 objects` in simulation (`Isaac Sim`).
    *   `2K real-world episodes` collected `without teleoperation` by employing a "real-world simulator" system for object state estimation.
        This benchmark provides a standardized platform for evaluating `VLA` policies on moving objects across interaction, perception, and generalization dimensions.
*   **Demonstrated Superior Performance:** Extensive evaluations on the `DOM` benchmark, including `16 real-robot tasks` across multiple embodiments (Franka Emika Panda and AgileX PiPER), show `DynamicVLA` achieves remarkable improvements in response speed, perception, and generalization compared to existing `VLA` baselines. It significantly reduces `task completion time` and increases `success rates` in dynamic scenarios.

    In essence, `DynamicVLA` provides a unified framework that tackles the core problem of `latency-induced failure` in dynamic object manipulation by offering a lightweight model architecture combined with intelligent inference and action execution strategies, supported by a novel, large-scale dataset for training and evaluation.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### Vision-Language-Action (VLA) Models
`Vision-Language-Action (VLA)` models are a cutting-edge class of artificial intelligence models that extend the capabilities of `Vision-Language Models (VLMs)` to robotic control.
*   **Vision-Language Models (VLMs):** These models are trained on vast datasets of images paired with descriptive text. They learn to understand and generate content that combines visual and linguistic information. For example, a `VLM` can describe the contents of an image ("a cat sitting on a mat") or answer questions about an image ("What color is the cat?" -> "Orange"). Popular examples include `CLIP`, `Flamingo`, and `LLaVA`.
*   **VLA Extension:** `VLA` models build upon `VLMs` by adding an `action generation` component. This allows them to not only perceive the environment visually and understand natural language instructions but also to translate that understanding into executable actions for a robot. The goal is to enable robots to understand high-level commands (e.g., "pick up the red apple") and perform complex manipulation tasks by predicting a sequence of low-level robot joint commands or end-effector poses.
*   **How they work:** Typically, a `VLA` model takes multimodal inputs—visual observations (e.g., camera images), language instructions (e.g., "grasp the bottle"), and robot proprioceptive state (e.g., current joint angles or end-effector pose)—and outputs a sequence of actions or a policy that dictates robot movement. These actions can be `6DoF end-effector poses` (position and orientation) or joint velocities/torques.

### Diffusion Models
`Diffusion models` are a class of `generative models` that have gained prominence in recent years, particularly for image and video synthesis. They are also increasingly being adapted for `sequential data generation`, including robot actions.
*   **Core Idea:** `Diffusion models` work by learning to reverse a `diffusion process`. In the forward diffusion process, noise is gradually added to data until it becomes pure random noise. The reverse process involves learning to progressively `denoise` the data, starting from random noise and gradually transforming it back into a coherent data sample (e.g., a clear image or a meaningful action sequence).
*   **Action Generation:** When applied to action generation, a `diffusion model` learns to predict a sequence of robot actions. Instead of directly predicting the actions, it learns to `denoise` a noisy version of an action sequence, conditioned on the robot's current state, visual observations, and language instructions. This allows for diverse and high-quality action generation, as the model explores a distribution of possible actions rather than just a single deterministic output.
*   **Flow Matching:** A related technique, `Flow Matching`, is used in `DynamicVLA`. It is an alternative to standard `diffusion models` for learning continuous-time generative models. Instead of learning to denoise, `Flow Matching` learns to predict a `vector field` that transports samples from a simple base distribution (e.g., Gaussian noise) to the target data distribution (e.g., robot action sequences). This can offer advantages in training stability and sample quality.

### Vision Encoders (Convolutional vs. Transformer-based)
`Vision encoders` are components within `VLA` models responsible for processing visual input (images or video frames) and extracting meaningful features.
*   **Convolutional Vision Encoders (e.g., FastViT):** These encoders are built primarily using `convolutional neural networks (CNNs)`. `CNNs` are adept at capturing local spatial hierarchies and patterns in images through filters that scan across the input.
    *   **Advantages:** Typically efficient in terms of computation and memory for image processing, especially for high-resolution inputs. They perform `spatial compression` effectively by reducing the resolution of feature maps while increasing channel depth. `FastViT` specifically combines convolutional layers for early spatial compression with attention mechanisms in later stages.
    *   **Disadvantages:** Traditionally, `CNNs` have a limited `global receptive field` without significant architectural modifications, making it harder to capture long-range dependencies in an image.
*   **Transformer-based Vision Encoders (e.g., ViT, Swin Transformer):** These encoders adapt the `Transformer architecture`, originally developed for natural language processing, to vision tasks. They treat image patches as sequences of tokens and use `self-attention mechanisms` to model relationships between these tokens.
    *   **Advantages:** Excellent at capturing `global long-range dependencies` across an image, leading to powerful representational capacity.
    *   **Disadvantages:** Can be computationally expensive, especially for high-resolution images, as the `self-attention` mechanism scales quadratically with the number of tokens (image patches). This leads to `quadratic token growth`, which can be a bottleneck for processing `multiframe visual inputs` (multiple images over time).

### Real-time Control and Inference Latency
*   **Real-time Control:** In robotics, `real-time control` means that the robot system must respond to sensory inputs and execute actions within strict time limits. If the system is too slow, it might miss crucial events, leading to instability or failure. For dynamic manipulation, this implies a need for very low `latency`.
*   **Inference Latency:** This is the delay between when a `VLA` model receives an observation (e.g., a camera frame) and when it produces its predicted action. This delay includes the time taken for:
    1.  **Perception:** Encoding visual and other sensory inputs.
    2.  **Reasoning:** Processing multimodal features through the `VLM` backbone.
    3.  **Action Generation:** Producing the action sequence via the action expert.
        In dynamic tasks, `inference latency` can cause a `temporal misalignment` where the predicted action is based on an outdated perception of the environment, making it ineffective for the current, rapidly changing object state.

### Sim-to-Real Gap
The `sim-to-real gap` refers to the challenge of transferring policies or models trained in simulated environments to real-world robots.
*   **Simulation Advantages:** Simulations offer scalability (easy to generate vast amounts of data), safety (no risk to physical hardware), and access to perfect state information (ground-truth object poses and velocities).
*   **Real-world Challenges:** The real world introduces complexities not perfectly captured in simulation, such as sensor noise, material properties variations, unmodeled physics, lighting changes, and calibration errors.
*   **Bridging the Gap:** Researchers often use techniques like `domain randomization` (varying parameters in simulation to increase robustness) or `real-world fine-tuning` to help models generalize better from simulation to reality. The `DOM` benchmark specifically aims to bridge this gap by collecting real-world data in a structured, automated way.

## 3.2. Previous Works

The paper discusses related work in three main categories: `Vision-Language-Action Models`, `Robot Learning Datasets`, and `Robot Dynamic Manipulation`.

### Vision-Language-Action Models
Previous `VLA` models largely fall into several categories:
*   **Transformer-based methods** [57, 7]: These use `Transformers` to model sequences of states, actions, and rewards. `RT-1` [7] is a notable example, demonstrating real-world control at scale.
*   **LLM/VLM-based methods** [60, 17]: These treat `VLA` tasks as sequence-to-sequence problems, often leveraging large pretrained `LLMs` or `VLMs` for action generation. `OpenVLA` [17] is an open-source `VLA` model that emphasizes this approach.
*   **Diffusion-based methods** [9, 32]: These model policies as `denoising diffusion models`. `Diffusion Policy` [9] is a prominent example, learning visuomotor policies through action diffusion.
*   **Combined LLM/Diffusion methods** [14, 6]: These combine `LLMs` for representation learning with `diffusion models` for action generation, such as $\pi_0$ [6].
*   **Video generation with inverse kinematics methods** [53, 49, 37]: These approaches generate motion sequences and then convert them into robot actions using `inverse kinematics`.

    **Key limitation identified by DynamicVLA:** Most existing `VLA` models suffer from slow inference speeds. This makes them unsuitable for tasks requiring `precise` or `rapid execution`, especially in dynamic environments where latency leads to `temporal misalignment`. The paper mentions `SmolVLA` [38] and `VLA-Adapter-Pro` [46] as efforts to improve efficiency by reducing model size and increasing throughput, but `DynamicVLA` argues these still don't fully address the `inference delay` in dynamic settings.

### Robot Learning Datasets
*   **Real-world datasets** [28, 45, 35, 50]: Provide high-fidelity interactions but are expensive and difficult to scale. `BridgeData V2` [45] and `Open X-Embodiment` [35] are examples of efforts to create large-scale real-world datasets.
*   **Simulated datasets** [30, 24, 20, 21, 33]: Offer scalability but suffer from the `sim-to-real gap`. `CALVIN` [30] and `LIBERO` [24] focus on long-horizon or language-conditioned tasks.
*   **Generative models for data** [34, 47, 53, 48]: Attempt to create interactive data but are often constrained by artifacts or memory.

    **Key limitation identified by DynamicVLA:** Current datasets overwhelmingly focus on `static scenes` and lack a large-scale foundation for `dynamic object manipulation`. This absence is a major barrier to developing and evaluating `VLA` models capable of handling moving objects. Teleoperation, a common method for data collection, is explicitly deemed ineffective for fast-moving objects due to human reaction limits.

### Robot Dynamic Manipulation
Historically, robotic manipulation has been studied predominantly in `static settings`.
*   **Task-specific/Predictable Motion:** Existing methods for moving objects tend to be `task-specific` (e.g., throwing, soccer, table tennis [54, 19, 10]) or rely on `predictable motion` (e.g., conveyor-belt-like scenarios as in `DBC-TFP` [56] and `GEM` [22]). These often use reactive control and handcrafted perception pipelines in structured environments.
*   **Tolerant Interactions:** Concurrent `VLA` methods like `RDT2` [43], `RTVLA` [27], and `VLASH` [39] have shown real-time interaction with fast-moving targets. However, `DynamicVLA` notes that these tasks often permit `large contact margins` (e.g., hitting a ping-pong ball with a paddle) and do not require the `precise 6DoF manipulation` needed for general dynamic object manipulation.

    **Key limitation identified by DynamicVLA:** `Open-ended dynamic manipulation` involving `uncertain motion`, `precise contact`, and `tight perception-action alignment` remains largely unsolved.

## 3.3. Technological Evolution
The field of robot manipulation has evolved from:
1.  **Early Robotics (Rule-based/Model-based):** Primarily focused on structured environments with pre-programmed trajectories or precise models of the environment. Dynamic tasks were often handled by reactive control with specialized sensors and hand-tuned controllers.
2.  **Learning-based Robotics (Deep Reinforcement Learning, Imitation Learning):** Moved towards learning policies from data, often in simulation, to handle more complex or uncertain static tasks. This led to increased task diversity.
3.  **Vision-Language Models (VLMs) and Foundation Models:** The advent of large language models (`LLMs`) and `VLMs` enabled robots to understand natural language instructions and generalize across tasks and objects more effectively, leading to the development of `VLA` models for static manipulation.
4.  **Addressing Efficiency and Generalization in VLAs:** Recent efforts focused on making `VLA` models smaller and faster (`SmolVLA`, `VLA-Adapter-Pro`) and improving their generalization across embodiments (`Open X-Embodiment`, `RT-X models`).

    `DynamicVLA` positions itself at the forefront of this evolution by pushing `VLA` models into the uncharted territory of `dynamic object manipulation`. It directly addresses the critical `latency` issue that arises when applying powerful but often slow `VLA` models to real-time, rapidly changing environments, a problem largely ignored by prior `VLA` research for manipulation. It also fills a crucial `data gap` by creating a benchmark specifically for dynamic tasks.

## 3.4. Differentiation Analysis
`DynamicVLA` differentiates itself from previous works through a multi-faceted approach:

*   **Explicit Latency Management:** Unlike most `VLA` models that assume static environments or tolerate latency, `DynamicVLA` explicitly designs its architecture and execution strategy to minimize and manage `inference delay`. This is achieved through:
    *   **Compact Model Size:** A `0.4B` parameter model, significantly smaller than many `VLM`-based `VLAs` (which can be `3B-7B`), to ensure high-frequency reasoning.
    *   **Convolutional Vision Encoder (`FastViT`):** Chosen over transformer-based encoders for `spatially efficient` and `structurally faithful encoding` without the `quadratic token growth` bottleneck of transformers, leading to faster processing of multi-frame inputs.
    *   **Continuous Inference (CI):** A novel pipelined execution scheme that `overlaps reasoning and action execution`, eliminating `inter-chunk waiting` that plagues traditional serial execution in `VLA`s.
    *   **Latent-aware Action Streaming (LAAS):** An intelligent execution mechanism that `discards outdated actions` and `prioritizes the most recent predictions`, thus restoring temporal alignment between perception and action despite inherent inference delays. This is a core innovation for closed-loop adaptation.

*   **Dedicated Dynamic Manipulation Benchmark (`DOM`):** Previous datasets primarily focus on static manipulation. `DynamicVLA` introduces the first large-scale benchmark specifically for `dynamic object manipulation`, which is critical for standardized evaluation and future research.
    *   **Automated Data Collection:** `DOM` features fully automated data collection pipelines for both simulation (`200K episodes`) and the real world (`2K episodes`), crucially `without teleoperation`. This addresses the inadequacy of human teleoperation for fast-moving objects and provides high-quality, diverse data.
    *   **Multi-embodiment Support:** The benchmark is validated across multiple robot embodiments (Franka Emika Panda, AgileX PiPER), promoting generalizability.

*   **Focus on Precise 6DoF Dynamic Control:** While some `VLA`s (e.g., `RDT2`, `RTVLA`, `VLASH`) have shown interaction with fast-moving targets, `DynamicVLA` emphasizes `open-ended dynamic manipulation` requiring `uncertain motion`, `precise contact`, and `tight perception-action alignment` for `6DoF control`, which is a harder problem than tasks allowing large contact margins (like hitting a ball with a paddle).

    In summary, `DynamicVLA` directly targets the `temporal challenges` in dynamic robotics, which existing `VLA` models and datasets have largely overlooked, providing both a specialized efficient model and a foundational benchmark to push this frontier forward.

# 4. Methodology

## 4.1. Principles
The core principle behind `DynamicVLA` is to address the fundamental challenge of `temporal misalignment` between perception and action execution in `dynamic object manipulation`. This misalignment, caused by `inference latency`, leads to robot actions being based on outdated information, making them ineffective in rapidly changing environments. `DynamicVLA` tackles this by:
1.  **Minimizing Inference Latency:** Designing a `compact` and `efficient` `Vision-Language-Action (VLA)` model capable of `high-frequency reasoning`.
2.  **Overlapping Reasoning and Execution:** Introducing a `pipelined execution scheme` to ensure a continuous action stream and reduce waiting times.
3.  **Restoring Temporal Alignment:** Implementing a mechanism to ensure that executed actions are always based on the most `recent` and `relevant` environmental observations, despite inherent delays.

    The theoretical basis draws from the need for tightly coupled `sense-plan-act` loops in dynamic systems, where any delay in the loop can lead to instability or failure. By integrating `temporal reasoning` and `closed-loop adaptation`, `DynamicVLA` aims to achieve robust and responsive control for moving objects.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Problem Formulation
The problem is defined as `dynamic object manipulation`, where a robot must interact with objects whose states continuously change during the processes of perception, reasoning, and execution.
At each `time step` $t$:
*   The `VLA` model, denoted as $\mathcal{M}$, receives a `temporal window` of visual observations $\mathbf{O}_t = \{ \mathbf{o}_{t-k}, \dots, \mathbf{o}_t \}$. Here, $\mathbf{o}_t$ represents the visual observation at time $t$, and $k$ determines the length of the historical context.
*   A `language instruction` $\mathbf{L}_t$ (e.g., "grasp the red box").
*   The robot's `proprioceptive state` $\mathbf{P}_t$ (e.g., current joint positions, end-effector pose).
*   Based on these inputs, the model predicts an `action sequence` $\mathbf{A}_t = \{ \mathbf{a}_t, \dots, \mathbf{a}_{t+n} \}$, where $\mathbf{a}_t$ is the action for time $t$, and $n$ is the `action horizon` (the number of future actions predicted).
    This relationship is formally expressed as:
\$
\mathbf{A}_t = \mathcal{M}(\mathbf{O}_t, \mathbf{L}_t, \mathbf{P}_t)
\$
A crucial aspect of this problem is the `latent object state` $\mathbf{s}_t$, which describes the object's `6D pose` (position and orientation) and `motion` (linear and angular velocities). Importantly, this object motion does not pause during the model's inference. If the model starts reasoning over $\mathbf{O}_t$ at time $t$, by the time its predictions become available at $t+m$ (where $m$ is the `inference latency`), the object's state will have evolved from $\mathbf{s}_t$ to $\mathbf{s}_{t+m}$. This dynamic evolution leads to a potential `misalignment between perception and execution`.

### 4.2.2. The DynamicVLA Architecture
The `DynamicVLA` architecture (Figure 2a from the original paper) is designed to be compact and efficient to enable `fast multimodal inference` and minimize `inference latency`.

The following figure (Figure 2 from the original paper) shows the architecture of DynamicVLA, its Continuous Inference, and Latent-aware Action Streaming:

![该图像是示意图，展示了DynamicVLA框架的架构与关键设计。图中包括三个主要部分：第一部分(a)显示了DynamicVLA架构，其中包含SmolLM2-360M模型和Action Expert；第二部分(b)介绍了连续推理的流程，强调推理循环与执行循环的关系；第三部分(c)展示了潜在感知动作流，描述了输入流与动作流的关系。这些设计旨在实现动态物体的高效操控和适应。](images/2.jpg)  
*该图像是示意图，展示了DynamicVLA框架的架构与关键设计。图中包括三个主要部分：第一部分(a)显示了DynamicVLA架构，其中包含SmolLM2-360M模型和Action Expert；第二部分(b)介绍了连续推理的流程，强调推理循环与执行循环的关系；第三部分(c)展示了潜在感知动作流，描述了输入流与动作流的关系。这些设计旨在实现动态物体的高效操控和适应。*

VLM Description: The image is a schematic diagram illustrating the architecture and key design of the DynamicVLA framework. It includes three main sections: the first part (a) shows the DynamicVLA architecture, featuring the SmolLM2-360M model and Action Expert; the second part (b) introduces the process of continuous reasoning, highlighting the relationship between the reasoning cycle and the execution cycle; the third part (c) depicts the latent perception action flow, describing the relationship between the input flow and action flow. These designs are aimed at achieving efficient control and adaptation of dynamic objects.

#### 4.2.2.1. Vision-Language Backbone
The backbone consists of a vision encoder and a language model, forming a `Vision-Language Model (VLM)`:
*   **Language Backbone:** `SmolLM2-360M` [3] is adopted, resulting in a compact overall model size of `0.4 billion` parameters. This model is a smaller, more efficient language model. Following `SmolVLA` [38], the language backbone is `truncated` to its first `16 transformer layers`. This significantly reduces `inference latency` while maintaining sufficient `multimodal reasoning` capabilities.
*   **Vision Encoder:** Unlike many existing `VLM`s that rely on transformer-based vision encoders (which can suffer from `quadratic token growth` with multi-frame inputs), `DynamicVLA` employs a `convolutional vision encoder`, specifically `FastViT` [44]. `FastViT` is chosen because it performs `efficient spatial compression` and `preserves structural fidelity` in visual representations. This enables faster processing of visual inputs crucial for dynamic manipulation.
    *   **Multi-frame Visual Inputs:** The visual observations $\mathbf{O}_t$ consist of a temporal window of images (e.g., $\mathbf{o}_{t-2}, \mathbf{o}_t$). These are concatenated (e.g., channel-wise if from multiple views or time steps) and fed into `FastViT`.
*   **Multi-modal Fusion and Projection (Appendix A):** Lightweight `linear projections` are used to align representations across different modules:
    1.  `Embedding robot states`: The 32-dimensional robot `proprioceptive state` $\mathbf{P}_t$ (Cartesian position, orientation, gripper state, with zero padding) is linearly projected into the `language embedding space` (a 960-dimensional state token).
    2.  `Adapting action representations`: Aligning action features for the diffusion-based action expert.
    3.  `Matching output dimensions`: Ensuring compatibility between the `VLM` backbone's output and the action expert's input.
        All visual tokens (from `FastViT`), language tokens (from `SmolLM2-360M`), and the projected state token are concatenated and processed jointly by the language backbone. The language backbone outputs `key-value representations` for these tokens, which are cached and reused in subsequent inference cycles to save computation.

#### 4.2.2.2. Diffusion-Based Action Expert
The `action expert` $\mathcal{E}_\theta$ is responsible for predicting the action chunk $\mathbf{A}_t$, conditioned on the `multimodal features` $\mathbf{f}_t$ produced by the `VLM` backbone.
*   **Instantiation:** Following modern diffusion-style action modeling [23, 12], $\mathcal{E}_\theta$ is instantiated as a `conditional Flow Matching Transformer` [6]. This means it learns to predict actions by modeling a continuous flow that transforms noise into action sequences.
*   **Architecture:** The action expert is a lightweight transformer, copied from the language backbone and also truncated to its first `16 layers`. It predicts an `action chunk` with a horizon of $n=20$ actions. Each action is a 32-dimensional vector representing the `end-effector pose` and `gripper state`.
*   **Training Objective:** The action expert is trained using the `Flow Matching` objective:
    \$
    \ell ^ { \tau } ( \theta ) = \mathbb { E } _ { p ( \mathbf { A } _ { t } \mid \mathbf { f } _ { t } ) , q ( \mathbf { A } _ { t } ^ { \tau } \mid \mathbf { A } _ { t } ) } \left[ \left\| \mathcal { E } _ { \theta } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { O } _ { t } ) - \mathbf { u } ( \mathbf { A } _ { t } ^ { \tau } \mid \mathbf { A } _ { t } ) \right\| \right]
    \$
    Where:
    *   $\ell^{\tau}(\theta)$: The `loss function` for the action expert, parameterized by $\theta$. The superscript $\tau \in [0, 1]$ denotes `flow matching timesteps`, representing a continuous interpolation between noise and data.
    *   $\mathbb{E}$: Denotes the `expectation` over the data distribution.
    *   $p(\mathbf{A}_t \mid \mathbf{f}_t)$: The true distribution of action sequences $\mathbf{A}_t$ conditioned on the `VLM` features $\mathbf{f}_t$.
    *   $q(\mathbf{A}_t^\tau \mid \mathbf{A}_t)$: A simple `conditional distribution` that describes how a noisy action chunk $\mathbf{A}_t^\tau$ is generated from a true action chunk $\mathbf{A}_t$ at timestep $\tau$. It is defined as a Gaussian distribution: $q(\mathbf{A}_t^\tau | \mathbf{A}_t) = \mathcal{N}(\tau \mathbf{A}_t, (1-\tau)\mathbf{I})$. This means $\mathbf{A}_t^\tau$ is a linear interpolation between $\mathbf{A}_t$ and noise, where $\tau$ controls the interpolation factor.
    *   $\mathbf{A}_t^\tau$: A `noisy action chunk` at flow matching timestep $\tau$. It is generated by interpolating between the true action chunk $\mathbf{A}_t$ and a random noise vector $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: $\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1 - \tau) \epsilon$.
    *   $\mathbf{O}_t$: The `visual observations` at time $t$. Although $\mathbf{f}_t$ (VLM features) is the primary conditioning for $\mathcal{E}_\theta$, the formulation explicitly shows $\mathbf{O}_t$ in the expert's input, implying the multimodal context is derived from it.
    *   $\mathbf{u}(\mathbf{A}_t^\tau \mid \mathbf{A}_t)$: The `denoising vector field` that the expert is learning to match. In this context, it's given as $\epsilon - \mathbf{A}_t$, which represents the direction to move from $\mathbf{A}_t^\tau$ to remove the noise and recover $\mathbf{A}_t$.
    *   $\mathbf{f}_t$: The `multimodal features` extracted by the `VLM` backbone from $\mathbf{O}_t$, $\mathbf{L}_t$, and $\mathbf{P}_t$.
    *   $\epsilon$: A sample from a `standard normal distribution` $\mathcal{N}(0, \mathbf{I})$, representing pure Gaussian noise.
    *   $\mathbf{I}$: The `identity matrix`, indicating that the noise is isotropic (same variance in all dimensions).
        Under this objective, the action expert $\mathcal{E}_\theta(\mathbf{A}_t^\tau, \mathbf{O}_t)$ learns to predict the `denoising vector field` required to transform a noisy action sequence $\mathbf{A}_t^\tau$ back into the clean, desired action sequence $\mathbf{A}_t$, conditioned on the current environment state. During inference, the expert starts with pure noise and iteratively applies the learned vector field to generate a coherent action sequence.

### 4.2.3. Continuous Inference (CI)
Traditional `VLA` models operate in a serial manner: an inference cycle is triggered, the action sequence is predicted, and only after this sequence is fully executed is the next inference cycle initiated. This introduces `inter-chunk waiting`, where the robot idles, leading to degraded responsiveness, especially problematic in dynamic settings.
`Continuous Inference (CI)` (Figure 2b from the original paper) is a `pipelined execution scheme` designed to eliminate this waiting.
*   **Mechanism:** Inference cycles are triggered `asynchronously` as soon as the previous inference completes, `independent of whether the previously predicted action sequence has been fully executed`.
*   **Timing:** Let $m$ be the `inference delay` (number of timesteps between the start and completion of an inference cycle). Inference completes at timesteps $t, t+m, t+2m, \ldots$. For simplicity, $m$ is assumed constant, though it can vary.
*   **Execution Flow:** During execution, actions from the currently available action sequence (e.g., $\mathbf{A}_t$) are executed continuously. Simultaneously, the next action sequence (e.g., $\mathbf{A}_{t+m}$) is being inferred.
*   **Condition:** This scheme assumes that the `action horizon` $n$ (length of the predicted action sequence) is `greater than the inference delay` $m$ ($n > m$). This ensures that a new action sequence becomes available before the execution of the current sequence is complete, preventing execution from blocking on inference completion and thereby eliminating `inter-chunk waiting`.

### 4.2.4. Latent-aware Action Streaming (LAAS)
Even with `Continuous Inference`, the `inference delay` $m$ still introduces `temporal misalignment` between the predicted actions and the environment's actual evolving state. `Latent-aware Action Streaming (LAAS)` (Figure 2c from the original paper) is an `explicit execution strategy` designed to resolve this misalignment.
*   **Perception-Execution Gap:** When inference for $\mathbf{A}_t$ starts at time $t$, the predicted actions only become available at $t+m$. By this time, the environment has evolved to $\mathbf{O}_{t+m}$. This means actions $\mathbf{a}_t, \ldots, \mathbf{a}_{t+m-1}$ (the first $m$ actions of $\mathbf{A}_t$) are `outdated` because they were predicted based on the older observation $\mathbf{O}_t$.
*   **Conflicts Between Overlapping Action Chunks:** `Continuous Inference` allows new action sequences (e.g., $\mathbf{A}_{t+m}$) to be generated before the execution of previous ones (e.g., $\mathbf{A}_t$) is complete. This results in multiple candidate actions for the same future execution timestep.
*   **Resolution Strategy:** `LAAS` addresses these issues by:
    1.  **Discarding Outdated Actions:** Actions in $\mathbf{A}_t$ corresponding to timesteps `earlier than` $t+m$ (i.e., $\{\mathbf{a}_t, \ldots, \mathbf{a}_{t+m-1}\}$) are `discarded`. Execution then proceeds with the subsequence $\{\mathbf{a}_{t+m}, \ldots, \mathbf{a}_{t+n}\}$ from the available chunk.
    2.  **Prioritizing Newer Predictions:** For timesteps where `multiple action chunks overlap` (e.g., an action $\mathbf{a}_j$ from $\mathbf{A}_t$ and an action $\mathbf{a}'_j$ from $\mathbf{A}_{t+m}$ both correspond to the same real-world time $j$), actions from the `newer sequence` (e.g., $\mathbf{A}_{t+m}$) are `prioritized` and overwrite those from the older sequence ($\mathbf{A}_t$).
        This prioritization allows the robot to adapt promptly to the most recent environment state, ensuring `temporally consistent control` despite inference delays.

### 4.2.5. The Dynamic Object Manipulation (DOM) Benchmark
The `DOM` benchmark is introduced to fill the critical gap in large-scale data for dynamic object manipulation.

#### 4.2.5.1. Overview
*   **Purpose:** To provide a standardized, large-scale benchmark for evaluating robotic policies on moving objects.
*   **Data Scale:** `200K synthetic episodes` and `2K real-world episodes`.
*   **Collection Method:** Fully `automated data-collection pipelines` in both simulation and the real world, avoiding teleoperation.
*   **Evaluation Dimensions:** Organized along `structured interaction`, `perception`, and `generalization` dimensions for consistent and comparable evaluation.

#### 4.2.5.2. Benchmark Dimensions (Figure 1c from the original paper)
The `DOM` benchmark evaluates policies across three main dimensions:
*   **Interaction:** Measures effective response to evolving object motion.
    1.  `Closed-loop reactivity (CR)`: How quickly the robot adjusts to different object speeds.
    2.  `Dynamic adaptation (DA)`: Handling abrupt changes in motion (e.g., direction shifts, disturbances).
    3.  `Long-horizon sequencing (LS)`: Maintaining coherent behavior over extended interactions with multiple moving objects.
*   **Perception:** Measures how well a policy perceives and grounds visual and linguistic cues in dynamic settings.
    1.  `Visual understanding (VU)`: Distinguishing objects with similar shapes, textures, materials.
    2.  `Spatial reasoning (SR)`: Inferring object positions and relative arrangements in cluttered or changing scenes.
    3.  `Motion perception (MP)`: Accurately interpreting object motion cues (speed, direction).
*   **Generalization:** Measures robustness to distribution shifts beyond training conditions.
    1.  `Visual generalization (VG)`: Adaptation to unseen shapes, appearances, scene layouts.
    2.  `Motion generalization (MG)`: Handling new speed ranges, altered friction, novel trajectory patterns.
    3.  `Disturbance Robustness (DR)`: Maintaining stable behavior under external perturbations (pushes, collisions, sensor noise).

#### 4.2.5.3. Simulation Data Collection
A high-throughput pipeline is built in `Isaac Sim` [31] for scalable data generation.

The following figure (Figure 3 from the original paper) illustrates the automated data collection pipeline:

![该图像是示意图，展示了动态物体操作的环境设置和状态机控制流程。图中包括了206个对象和2824个场景的信息，以及使用Isaac Sim和实际场景的相关数据采集。整个流程分为四个步骤：接近对象、抓取并提升、接近目标并放置，以及重置。](images/3.jpg)  
*该图像是示意图，展示了动态物体操作的环境设置和状态机控制流程。图中包括了206个对象和2824个场景的信息，以及使用Isaac Sim和实际场景的相关数据采集。整个流程分为四个步骤：接近对象、抓取并提升、接近目标并放置，以及重置。*

VLM Description: The image is a schematic diagram illustrating the setup and state machine control process for dynamic object manipulation. It includes information on 206 objects and 2824 scenes, along with data collection from both the Isaac Sim and real-world scenarios. The entire process is divided into four steps: approach object, grasp and lift, approach target and place, and reset.

*   **Objects and Dynamics:**
    *   `206 everyday objects` from `Objeverse` [11] (fruits, vegetables, containers, etc.), with texture augmentation.
    *   Object speeds sampled from $0-0.75 m/s$ (some static), friction coefficients from `0.5-1.5`. Multiple objects are placed for natural interactions.
*   **Scenes and Sensors:**
    *   `2.8K diverse 3D scenes` from `3D-FRONT` [13], curated for clean, flat tabletops.
    *   Three cameras: `two third-person views` (front and left at specific heights) and a `wrist-mounted camera`.
    *   All capture `RGB frames at 25 FPS` with `480x360 resolution`, matching Azure Kinect intrinsics.
    *   Randomized scene illumination (color temperature, light intensity, source positions).
*   **Object State Acquisition:**
    *   `Ground-truth 6D object states` (position, rotation, linear/angular velocity) are acquired from `Isaac Sim`'s physics engine at `25 Hz`. This provides noise-free, real-time motion cues.
*   **State Machine Controller:** A four-stage `closed-loop routine` consumes real-time 6D object pose, velocity, and static target pose to drive the robot:
    1.  `Approach Object`: Predict near-future object motion ($~0.2-0.3 s$) and position end-effector `10 cm` above predicted location with continuous updates.
    2.  `Grasp & Lift`: Descend, stabilize residual motion, secure grasp, and lift.
    3.  `Approach Target & Place`: Move to target pose (from target object's 6D geometry) and place accurately.
    4.  `Reset`: Return to home pose.
        This design generates reactive, prediction-informed trajectories for realistic dynamic manipulation episodes.

#### 4.2.5.4. Real-World Data Collection
To overcome teleoperation limitations and lack of ground truth, a "real-world simulator" pipeline is built.
*   **Environment Setup:**
    *   `25 physical household objects` with multiple objects per episode.
    *   Two synchronized `third-person RGB cameras` (`Azure Kinect DK`) at front and side viewpoints.
    *   A `wrist-mounted RealSense D435i`.
    *   Geometry matches simulation.
*   **Object State Acquisition:** Replicates the simulator's 6D state interface:
    *   `EfficientTAM` [51] provides `per-view object masks` from synchronized third-person cameras.
    *   A `geometric triangulation step` recovers the `3D centroid`.
    *   `Linear and angular velocities` are obtained by `fitting motion over a short temporal window`.
    *   This produces a smooth, low-latency 6D state stream.
*   **State-machine Controller:** The `same four-stage controller` from simulation runs unchanged, consuming estimated 6D object states and target pose. This enables fast (`~10 s/episode`), teleoperation-free collection across Franka and PiPER robots.

### 4.2.6. The Training Scheme (Appendix B)
The training of `DynamicVLA` involves three stages:

#### 4.2.6.1. Pre-training Stage
*   **Objective:** To align visual and linguistic representations for the `Vision-Language Backbone`.
*   **Components:** The `convolutional visual encoder` (`FastViT`) and the `compact language model` (`SmolLM2-360M`) are initialized from their respective pretrained weights.
*   **Data:** Large-scale `vision-language pre-training` is performed using `150 million English image-text pairs` sampled from `COYO-700M` [8]. This stage ensures the `VLM` backbone has a strong foundation in multimodal understanding.

#### 4.2.6.2. Mid-training Stage
*   **Objective:** Train the full `VLA` model on the synthetic `Dynamic Object Manipulation (DOM)` dataset.
*   **Observations:** The model uses a `sparse temporal observation window` $\mathbf{O}_t = \{ \mathbf{o}_{t-2}, \mathbf{o}_t \}$, meaning it takes frames from the current time $t$ and two timesteps prior `t-2`. With two views per timestep (wrist-mounted and fixed third-person), this results in four images per input step, concatenated channel-wise. This sparse context is designed for `implicit object velocity perception`.
*   **Training Process:** `DynamicVLA` is optimized using `minibatches` of randomly sampled episode timesteps from shuffled manipulation demonstrations. For each minibatch, the model is trained on tuples $(\mathbf{O}_t, \mathbf{L}_t, \mathbf{P}_t)$, and the action expert is trained to `denoise` a noisy action chunk $\mathbf{A}_t^\tau$ using the `Flow Matching objective` defined in Eq. 1.

#### 4.2.6.3. Post-training Stage
*   **Objective:** Fine-tune the model on `robot-specific real-world demonstrations`.
*   **Process:** The same objective as in mid-training is used. This stage enables adaptation to specific new robot embodiments and sensing configurations, helping to bridge the `sim-to-real gap`.

### 4.2.7. Implementation Details (Appendix C)
*   **Training Hardware:** 32 NVIDIA A100 GPUs.
*   **Batch Size:** 40 per GPU.
*   **Optimizer:** `AdamW` with a learning rate of $1 \times 10^{-4}$, $\beta$ coefficients $(0.9, 0.95)$, $\epsilon = 1 \times 10^{-8}$, and weight decay of $1 \times 10^{-10}$.
*   **Learning Rate Schedule:** `Cosine learning rate schedule` with 1000 warm-up steps.
*   **Training Duration:** Approximately two weeks (2 days pre-training, 10 days mid-training, 2 days post-training).
*   **Inference Performance:** `DynamicVLA` requires `1.8GB of GPU memory` and runs at approximately `88 Hz` on an NVIDIA RTX A6000 GPU.

# 5. Experimental Setup

## 5.1. Datasets
The experiments are primarily conducted on the newly introduced `Dynamic Object Manipulation (DOM)` benchmark, created by the authors.

*   **Source and Scale:**
    *   **Synthetic Data:** `200K episodes` collected in `Isaac Sim` [31]. This dataset spans `2.8K diverse 3D scenes` (from `3D-FRONT` [13]) and features `206 everyday objects` (from `Objeverse` [11]).
    *   **Real-World Data:** `2K episodes` collected from scratch in real-world settings without teleoperation. This involves `25 physical household objects`.
*   **Characteristics and Domain:**
    *   **Dynamic Nature:** The core characteristic is that objects are continuously in motion, requiring real-time perception and adaptation.
    *   **Diversity:** The synthetic data features a wide range of objects (fruits, vegetables, containers, tools) and scene layouts, with randomized object speeds ($0-0.75 m/s$), friction coefficients (`0.5-1.5`), and illumination. This aims to improve `generalization`.
    *   **Multi-embodiment:** Data collection and evaluation are performed on two different robot arms: `Franka Emika Panda` and `AgileX PiPER`.
    *   **Ground Truth (Simulation):** In simulation, `ground-truth 6D object states` (position, rotation, linear/angular velocity) are available, which is crucial for training the state machine controller and evaluating policy performance.
    *   **Estimated States (Real-World):** In the real world, 6D object states are estimated using a perception system (`EfficientTAM` and geometric triangulation) to mimic the simulator's ground-truth interface, enabling the same state-machine controller to operate.
*   **Choice Rationale:** These datasets were chosen because existing robotic datasets overwhelmingly capture `static scenes` and lack the necessary scale and diversity for `dynamic object manipulation`. The `DOM` benchmark directly addresses this gap by providing a foundational dataset specifically designed for evaluating policies under moving objects, covering `interaction`, `perception`, and `generalization` challenges. The automated collection pipelines ensure reproducibility and efficiency, overcoming the limitations of teleoperation for fast dynamics.

## 5.2. Evaluation Metrics
All methods are evaluated using three primary metrics:

1.  **Success Rate (SR)**
    *   **Conceptual Definition:** This metric quantifies the effectiveness of a policy by measuring the proportion of trials where the robot successfully completes the instructed manipulation task. A trial is considered successful if the task is finished without dropping the object or exceeding a predefined time limit. It directly indicates the policy's reliability and capability to achieve task goals.
    *   **Mathematical Formula:**
        \$
        \mathrm{SR} = \frac{\text{Number of successful trials}}{\text{Total number of trials}} \times 100\%
        \$
    *   **Symbol Explanation:**
        *   $\mathrm{SR}$: Success Rate, expressed as a percentage.
        *   `Number of successful trials`: The count of experimental attempts where the robot successfully completed the manipulation task as per the instructions.
        *   `Total number of trials`: The total number of experimental attempts conducted for a given task.

2.  **Path Length (PL)**
    *   **Conceptual Definition:** This metric measures the total distance traveled by the robot's end-effector during the execution of a task. It serves as an indicator of the `efficiency` and `smoothness` of the robot's motion. A shorter path length generally suggests more optimized and direct movements, potentially implying better control and reduced wasted motion, though sometimes a longer path might be necessary for obstacle avoidance or more stable manipulation.
    *   **Mathematical Formula:** The paper does not provide a specific formula for `Path Length`, but it is a standard metric in robotics. It is typically calculated by summing the Euclidean distances between consecutive end-effector positions over the entire duration of the task.
        Let $P_t = (x_t, y_t, z_t)$ be the 3D position of the end-effector at time $t$. If the task spans $N$ timesteps, the path length is:
        \$
        \mathrm{PL} = \sum_{t=1}^{N-1} \sqrt{(x_{t+1}-x_t)^2 + (y_{t+1}-y_t)^2 + (z_{t+1}-z_t)^2}
        \$
    *   **Symbol Explanation:**
        *   $\mathrm{PL}$: Path Length, typically measured in meters (m).
        *   $P_t$: The 3D Cartesian coordinates of the robot's end-effector at time step $t$.
        *   $x_t, y_t, z_t$: The x, y, and z coordinates of the end-effector at time step $t$.
        *   $N$: The total number of timesteps (or discrete positions) recorded during the task execution.

3.  **Task Completion Time (Time)**
    *   **Conceptual Definition:** This metric measures the total duration, in seconds, from the moment an object's motion begins until the task officially terminates. Task termination can occur due to successful completion, a predefined timeout, or the object being dropped. This metric directly assesses the `responsiveness` and `efficiency` of the policy in completing dynamic tasks, where speed is often critical. Lower completion times indicate faster and more efficient execution.
    *   **Mathematical Formula:** The paper does not provide a specific formula, but it is a direct measurement of elapsed time.
        \$
        \mathrm{Time} = T_{\text{end}} - T_{\text{start}}
        \$
    *   **Symbol Explanation:**
        *   $\mathrm{Time}$: Task Completion Time, measured in seconds (s).
        *   $T_{\text{end}}$: The timestamp when the task officially terminates (either successfully, by timeout, or due to failure like an object drop).
        *   $T_{\text{start}}$: The timestamp marking the onset of the object's motion, which initiates the task.

## 5.3. Baselines
For comparative evaluation, `DynamicVLA` is benchmarked against several representative `VLA` baselines:

*   **Simulation Baselines:**
    *   `Diffusion Policy` [9]: A prominent `diffusion-based VLA` model that learns visuomotor policies via action diffusion.
    *   `OpenVLA-OFT` [18]: A variant of `OpenVLA` [17], an `open-source Vision-Language-Action model` focused on efficient fine-tuning.
    *   $\pi_0$ [6]: A `Vision-Language-Action flow model` designed for general robot control.
    *   $\pi_{0.5}$ [15]: An evolution of $\pi_0$, aiming for `open-world generalization` in `VLA` models.
    *   `SmolVLA` [38]: A `compact VLA` model designed for `affordable and efficient robotics`, similar in spirit to `DynamicVLA` in terms of efficiency.
    *   `GR00T-N1.5` [5]: An `open foundation model` (likely a large, generalist model) for humanoid robots, representing state-of-the-art generalist policies.
    *   `VLA-Adapter-Pro` [46]: An `effective paradigm for tiny-scale Vision-Language-Action models`, focusing on adaptation.
    *   `VLASH` [39]: A model emphasizing `real-time VLAs` via `future-state-aware asynchronous inference`, which is a latency-aware design.

*   **Real-World Baselines:**
    *   $\pi_{0.5}$ [15]
    *   `SmolVLA` [38]
    *   `VLASH` [39]
        These baselines were selected to cover a range of `VLA` approaches, including general-purpose models, lightweight and adaptation-based designs, and other latency-aware methods. All baselines are initialized from publicly available pretrained weights and adapted to the `DOM` benchmark using a consistent fine-tuning protocol to ensure fair comparison.

## 5.4. Execution Constraints
To ensure `safe real-world operation`, a critical `safety threshold` is imposed on the robot's workspace. If the predicted `end-effector position` by any policy exceeds these predefined bounds, the robot immediately `aborts` the current attempt and returns to a safe home pose. Such trials are automatically `marked as failure`. This constraint reflects practical safety considerations in real-world robotics.

## 5.5. Detailed Evaluation Setup (Appendix E)
Experiments are conducted in three environments: `Isaac Sim` with a `Franka Emika Panda` arm, a real-world `Franka` arm, and a real-world `AgileX PiPER` arm. Object motion is standardized across methods using a `secondary robot arm` following a fixed launching trajectory to ensure comparable motion patterns despite physical noise. Each real-world experiment is repeated `20 times`, and results are averaged.

### 5.5.1. Real-world Interaction Evaluation (Sec. V-B)
These tasks assess reactivity, adaptation, and long-horizon sequencing under dynamic object motion.
*   **Place the coffee can into the wooden box:** Track and grasp a `rolling Nescafé coffee can` and place it into a wooden box. Evaluates `closed-loop reactivity` to continuously moving targets.
*   **Place the conical bottle onto the frisbee:** Grasp a `conical roasted sesame bottle` with `curved trajectory` and place it onto a blue frisbee. Evaluates `closed-loop reactivity` under `non-linear motion`.
*   **Place the pickleball into the paper box:** Grasp a `moving pickleball` and place it into a paper box, where the ball is designed to `collide and deflect`. Evaluates `adaptive manipulation` under `contact-induced motion changes`.
*   **Place the ping pong ball inside the blue tape:** Grasp a `moving ping pong ball` and place it within a blue-taped region, where impacts with the tape `deflect trajectory`. Evaluates `adaptive placement` under `perturbed object motion`.
*   **Gather all ping pong balls into the paper box:** Continuously collect `ping pong balls that repeatedly appear` and place them into a paper box. Evaluates `long-horizon task sequencing` under `sustained dynamic inputs`.
*   **Gather all tennis balls into the red tape:** Continuously collect `tennis balls that repeatedly appear` and return them to a red-taped region. Evaluates `long-horizon planning and execution` in dynamic environments.

### 5.5.2. Real-world Perception Evaluation (Sec. V-C)
These tasks probe vision-language reasoning under dynamic manipulation, from visual recognition to spatial and motion perception.
*   **Place the tennis ball into the paper bowl:** Identify and grasp the `moving tennis ball` among `multiple simultaneously thrown objects` (tennis ball and pickleball), and place it into a paper bowl. Evaluates `object-level visual understanding` for identifying correct targets under dynamic motion.
*   **Place the tennis ball onto the blue-taped area:** Catch a `rolling tennis ball` and place it precisely within a `blue-taped region`, among `multiple visually similar tape markings` (red, blue, transparent). Evaluates `visually grounded target understanding` and `precise placement`.
*   **Place the cola can on the left wooden box:** Grasp a `moving cola can` and place it on a wooden box `located to its left`. Evaluates `spatial understanding` for target localization relative to the robot's viewpoint.
*   **Place the tennis ball on the right tape:** Grasp a `moving tennis ball` and place it on a tape `located to its right`. Evaluates `spatial understanding` for interpreting directionally specified targets.
*   **Place the slower ball into the paper bowl:** Grasp the `ping pong ball specified by its lower moving speed` and place it into the paper bowl. Evaluates `motion-based target understanding` based on movement speed.
*   **Place the faster-rolling can inside the frisbee:** Grasp the `cola can specified by its higher rolling speed` and place it inside the blue frisbee. Evaluates `motion-based target understanding` based on relative motion speed.

### 5.5.3. Real-world Generalization Evaluation (Sec. V-D)
These tasks assess robustness to distribution shifts in appearance, motion, and environmental perturbations.
*   **Place the plastic bottle into the wooden box:** Grasp a `rolling plastic bottle with an unseen appearance` and `regular curved trajectory`, place into wooden box. Evaluates `visual generalization` to unseen object appearances.
*   **Place the golf ball in the red tape:** Grasp a `rolling golf ball with an unseen appearance` and place it within a red-taped region. Evaluates `visual generalization` to unseen object instances.
*   **Place the potato into the wooden box:** Grasp a `moving potato whose motion follows irregular patterns` and place it into a wooden box. Evaluates `motion generalization` to irregular object dynamics.
*   **Place the green apple in the red tape:** Grasp a `moving green apple whose motion exhibits irregular and unpredictable patterns`, place onto a red-taped region. Evaluates `motion generalization` to irregular object trajectories.
    *(Note: Disturbance Robustness results are omitted for real-world due to difficulty in reliable reproduction and control of strong perturbations.)*

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental evaluations demonstrate that `DynamicVLA` achieves remarkable improvements in `response speed`, `perception`, and `generalization` for dynamic object manipulation compared to existing `VLA` baselines.

### 6.1.1. Dynamic Interaction and Reactivity
The `Interaction` dimension of the `DOM` benchmark evaluates `closed-loop reactivity (CR)`, `dynamic adaptation (DA)`, and `long-horizon sequencing (LS)`. These tasks progressively increase in difficulty, from reacting to varying speeds, to recovering from abrupt changes, and finally sustaining coordination over extended interactions.

The following are the results from Table I of the original paper:

<table><tr><td rowspan="2">Methods</td><td colspan="3">Interaction</td><td colspan="3">Perception</td><td colspan="3">Generalization</td><td colspan="3">Average</td></tr><tr><td>CR</td><td>DA</td><td>LS</td><td>VU</td><td>SR</td><td>MP</td><td>VG</td><td>MG</td><td>DR</td><td>SR ↑</td><td>Path Len ↓</td><td>Time ↓</td></tr><tr><td>Diffusion Policy [9]</td><td>0.50</td><td>0.50</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.50</td><td>0.00</td><td>0.38</td><td>1.34</td><td>10.89</td></tr><tr><td>OpenVLA-OFT [18]</td><td>3.50</td><td>0.50</td><td>0.50</td><td>0.00</td><td>1.50</td><td>0.50</td><td>3.50</td><td>2.00</td><td>0.00</td><td>1.33</td><td>1.08</td><td>10.83</td></tr><tr><td>\$π0 [6]</td><td>7.50</td><td>12.00</td><td>3.00</td><td>5.50</td><td>10.50</td><td>7.50</td><td>5.50</td><td>12.50</td><td>9.00</td><td>8.11</td><td>1.19</td><td>10.55</td></tr><tr><td>π0.5 [15]</td><td>9.50</td><td>17.50</td><td>3.50</td><td>5.00</td><td>12.50</td><td>9.00</td><td>5.00</td><td>19.50</td><td>18.00</td><td>11.06</td><td>1.28</td><td>10.62</td></tr><tr><td>SmolVLA [38]</td><td>18.50</td><td>17.50</td><td>5.50</td><td>1.50</td><td>14.50</td><td>11.50</td><td>14.50</td><td>13.50</td><td>17.00</td><td>12.67</td><td>1.30</td><td>10.65</td></tr><tr><td>GROOT-N1.5 [5]</td><td>10.50</td><td>12.00</td><td>4.00</td><td>9.50</td><td>13.50</td><td>14.00</td><td>14.50</td><td>19.50</td><td>20.00</td><td>13.05</td><td>1.29</td><td>10.56</td></tr><tr><td>VLA-Adapter-Pro [46]</td><td>21.00</td><td>15.50</td><td>6.00</td><td>6.50</td><td>16.50</td><td>10.50</td><td>15.00</td><td>18.50</td><td>13.00</td><td>13.61</td><td>1.51</td><td>9.98</td></tr><tr><td>VLASH [39]</td><td>9.00</td><td>20.50</td><td>7.50</td><td>6.50</td><td>7.50</td><td>12.00</td><td>7.00</td><td>21.00</td><td>20.00</td><td>12.33</td><td>1.27</td><td>10.60</td></tr><tr><td>DynamicVLA</td><td>60.50</td><td>38.50</td><td>40.50</td><td>51.50</td><td>48.00</td><td>33.50</td><td>59.50</td><td>65.00</td><td>26.50</td><td>47.06</td><td>2.50</td><td>8.53</td></tr></table>

*Table I: Dynamic Object Manipulation Simulation Benchmark Results. Average success rates (SR, %) are reported across overall average SR (%), path length (Path Len, meters), and task completion time (Time, seconds) are reported. Each method is evaluated over 1,800 trials (10 scenes × 9 dimensions × 20 trials). All baseline models are fine-tuned on the DOM dataset using their official implementations and released pretrained weights. Best results are highlighted in bold.*

As shown in Table I, `DynamicVLA` demonstrates significantly superior performance across all three interaction settings.
*   **Closed-loop reactivity (CR):** `DynamicVLA` achieves `60.50%` success, vastly outperforming the next best baseline (`VLA-Adapter-Pro` at `21.00%`). This indicates its strong ability to adjust quickly to objects moving at various speeds.
*   **Dynamic adaptation (DA):** `DynamicVLA` achieves `38.50%` success, again significantly higher than the strongest baseline (`VLASH` at `20.50%`). This highlights its capability to handle abrupt changes in object motion.
*   **Long-horizon sequencing (LS):** `DynamicVLA` shows `40.50%` success, a massive improvement over `VLASH` (`7.50%`). This underscores its capacity to maintain coordinated behavior over extended interactions with multiple dynamic objects.

    Overall, `DynamicVLA` outperforms the strongest baseline by $+188.1%$ (CR), $+87.8%$ (DA), and $+440.0%$ (LS) in these interaction settings. The general trend for prior `VLA`s is consistently low success rates under dynamic motion, validating the paper's premise that they struggle with latency and temporal misalignment.

The following figure (Figure 4 from the original paper) shows real-world interaction evaluation:

![该图像是图表，展示了不同模型在动态对象操控任务中的成功率。图中比较了DynamicVLA、VLASH、SmolVLA和π0.5四种模型在六个不同任务中的表现，突显了DynamicVLA在各任务中的优越性，成功率达71.6%。](images/4.jpg)  
*该图像是图表，展示了不同模型在动态对象操控任务中的成功率。图中比较了DynamicVLA、VLASH、SmolVLA和π0.5四种模型在六个不同任务中的表现，突显了DynamicVLA在各任务中的优越性，成功率达71.6%。*

VLM Description: The image is a chart that displays the success rates of different models in dynamic object manipulation tasks. It compares the performance of DynamicVLA, VLASH, SmolVLA, and π0.5 across six different tasks, highlighting the superiority of DynamicVLA with a success rate of 71.6%.

Real-world experiments (Figure 4) further corroborate these findings. Baselines frequently fail due to delayed reactions, stale action execution, or loss of coordination, while `DynamicVLA` reliably re-aligns perception and action under tight temporal constraints. For example, `DynamicVLA` achieves an average `71.6%` success rate in real-world interaction tasks, significantly higher than `VLASH` (`20.8%`), `SmolVLA` (`16.7%`), and $\pi_{0.5}$ (`15.0%`).

### 6.1.2. Multimodal Spatial-Temporal Reasoning
The `Perception` dimension evaluates `visual understanding (VU)`, `spatial reasoning (SR)`, and `motion perception (MP)`. This dimension demands timely and accurate interpretation of evolving spatial-temporal relationships.

As shown in Table I:
*   **Visual understanding (VU):** `DynamicVLA` achieves `51.50%` success, far surpassing `GROOT-N1.5` (`9.50%`). This demonstrates its strong ability to distinguish objects with similar appearances in dynamic scenes.
*   **Spatial reasoning (SR):** `DynamicVLA` achieves `48.00%` success, compared to `VLA-Adapter-Pro` (`16.50%`). This highlights its capability to infer object positions and relative arrangements in changing scenes.
*   **Motion perception (MP):** `DynamicVLA` achieves `33.50%` success, outperforming `GROOT-N1.5` (`14.00%`). This shows its accuracy in interpreting object motion cues like speed and direction.

    Performance of baselines degrades consistently across these tasks, especially in spatial and motion reasoning, as prior `VLA`s struggle with dynamic scenes and the trade-off between `VLM` capacity and `real-time latency`.

The following figure (Figure 5 from the original paper) shows real-world perception evaluation:

![该图像是图表，展示了不同模型在动态物体操作任务中的成功率。图表包含四种模型的性能比较：π0.5、SmolVLA、VLASH和DynamicVLA，涉及视觉理解、空间推理和运动感知三项任务。各模型的成功率以条形图形式呈现，DynamicVLA在多项任务中表现相对较好。](images/5.jpg)  
*该图像是图表，展示了不同模型在动态物体操作任务中的成功率。图表包含四种模型的性能比较：π0.5、SmolVLA、VLASH和DynamicVLA，涉及视觉理解、空间推理和运动感知三项任务。各模型的成功率以条形图形式呈现，DynamicVLA在多项任务中表现相对较好。*

VLM Description: The image is a chart that displays the success rates of different models in dynamic object manipulation tasks. The chart compares the performance of four models: π0.5, SmolVLA, VLASH, and DynamicVLA, across three tasks: visual understanding, spatial reasoning, and motion perception. The success rates of each model are presented in a bar graph, with DynamicVLA showing relatively better performance in several tasks.

Real-world perception evaluation (Figure 5) confirms `DynamicVLA`'s robustness. Baselines like `VLASH` achieve significantly lower success (`11.7%` average for Perception tasks) due to frequent `spatial-temporal misalignment`, whereas `DynamicVLA` reaches `51.9%` average success.

### 6.1.3. Generalization to Unseen Frontiers
The `Generalization` dimension assesses robustness to `distribution shifts` in appearance, motion, and environmental perturbations.

As shown in Table I:
*   **Visual generalization (VG):** `DynamicVLA` achieves `59.50%` success, far exceeding `VLA-Adapter-Pro` (`15.00%`). This indicates strong adaptation to unseen shapes, appearances, and scene layouts.
*   **Motion generalization (MG):** `DynamicVLA` achieves `65.00%` success, significantly higher than `VLASH` and `GROOT-N1.5` (both `21.00%`). This demonstrates its ability to handle new speed ranges, friction conditions, and trajectory patterns.
*   **Disturbance Robustness (DR):** `DynamicVLA` achieves `26.50%` success, which is an improvement over `GROOT-N1.5` (`20.00%`) and `VLASH` (`20.00%`), but the overall success rate for all models is lower in this challenging category. This suggests that strong perturbations remain a significant challenge.

    The following figure (Figure 6 from the original paper) shows real-world generation evaluation:

    ![Fig. 6: Real-world Generation Evaluation. We compare representative VLA models on four real-world dynamic manipulation tasks across Franka and PiPER, averaging success rates over 20 trials for each of three paired motionposition configurations, with object motion generated by a secondary robot arm.](images/6.jpg)
    *该图像是一个图表，展示了四种代表性的VLA模型在实际动态操作任务中的成功率比较。模型包括π0.5、SmolVLA、VLASH和DynamicVLA，数据展示了在视觉泛化与运动泛化任务中的表现。成功率以百分比形式显示，任务说明包括将瓶子、球以及其他物品放置于指定位置。*

Original caption: Fig. 6: Real-world Generation Evaluation. We compare representative VLA models on four real-world dynamic manipulation tasks across Franka and PiPER, averaging success rates over 20 trials for each of three paired motionposition configurations, with object motion generated by a secondary robot arm.

Similar trends are observed in real-world generalization experiments (Figure 6) for appearance and motion shifts. `DynamicVLA` achieves an average of `56.3%` success for `Visual Generalization` and `58.3%` for `Motion Generalization`, drastically outperforming `VLASH` (`10.0%` VG, `16.7%` MG), `SmolVLA` (`11.7%` VG, `13.3%` MG), and $\pi_{0.5}$ (`8.3%` VG, `10.0%` MG). The paper notes that robustness to environmental perturbations was challenging even for `DynamicVLA` in simulation, and real-world results for this specific aspect are omitted due to difficulties in reproduction.

### 6.1.4. Overall Averages
Overall, `DynamicVLA` achieves an average `Success Rate (SR)` of `47.06%` across all simulation tasks, which is more than `3.4 times` higher than the best baseline (`VLA-Adapter-Pro` at `13.61%`). It also achieves the lowest `Task Completion Time (Time)` at `8.53 seconds`, indicating faster and more efficient execution. Its `Path Length (PL)` is `2.50 meters`, which is higher than baselines, suggesting that robust dynamic control might require more intricate movements to adapt to changing conditions.

## 6.2. Ablation Studies
To understand the impact of `DynamicVLA`'s design choices, several ablation studies were conducted on the `DOM` benchmark.

The following are the results from Table II of the original paper:

<table><tr><td>Size</td><td>FViT</td><td></td><td>CI LAAS</td><td>SR (%) ↑</td><td>PL (m) ↓</td><td>Time (s) ↓</td></tr><tr><td>[1]</td><td>360M</td><td></td><td></td><td>30.27</td><td>2.77</td><td>9.86</td></tr><tr><td>[2] 360M</td><td></td><td></td><td>X J</td><td>36.11</td><td>1.77</td><td>9.51</td></tr><tr><td>[3] 360M</td><td>V</td><td>××&gt;</td><td>X</td><td>39.72</td><td>2.61</td><td>8.84</td></tr><tr><td>[4] 135M</td><td></td><td></td><td>✓</td><td>26.67</td><td>1.82</td><td>9.95</td></tr><tr><td>[5] 1.7B</td><td>V</td><td></td><td></td><td>24.33</td><td>1.77</td><td>9.91</td></tr><tr><td> 360M</td><td>×</td><td></td><td></td><td>28.89</td><td>1.86</td><td>9.89</td></tr><tr><td>[7] 360M</td><td>✓</td><td>V</td><td>✓</td><td>47.06</td><td>2.50</td><td>8.53</td></tr></table>

*Table II: Ablation of key design choices. The effects of LLM backbone size (Size), the use of FastViT as the vision encoder (FViT), Continuous Inference (CI), and Latent-aware Action Streaming (LAAS) are evaluated by reporting success rate (SR), path length (PL), and task completion time (Time) on the DOM benchmark. The final row corresponds to the DynamicVLA model configuration.*

### 6.2.1. Backbone Capacity
This study investigates the effect of `language model (LLM) capacity` by comparing `SmolLM2` backbones of different sizes (`135M`, `360M`, `1.7B`).
*   Comparing row `[4]` (`135M` LLM) with `[7]` (`360M` LLM): The `360M` model achieves `47.06% SR`, significantly higher than the `135M` model's `26.67% SR`. This indicates that reducing model size too much (to `135M`) limits reasoning capacity, leading to suboptimal action prediction despite potentially faster inference.
*   Comparing row `[5]` (`1.7B` LLM) with `[7]` (`360M` LLM): The `1.7B` model has a lower `SR` of `24.33%` compared to the `360M` model's `47.06%`. This shows that increasing model size too much (to `1.7B`) incurs higher `inference latency`, which degrades `closed-loop responsiveness` and results in lower success rates in dynamic scenarios.
*   **Conclusion:** The `360M` model strikes the best balance between `inference efficiency` and `model capacity`, yielding the highest overall performance in dynamic object manipulation (`SR 47.06%`).

### 6.2.2. Vision Encoder
This ablation compares `FastViT` (convolutional) with a `transformer-based vision encoder` (configured as in `SmolVLM` [29]).
*   Comparing row `[6]` (transformer-based, `SR 28.89%`) with `[7]` (`FastViT`, `SR 47.06%`): `FastViT` significantly outperforms the transformer-based encoder.
*   **Conclusion:** `FastViT` is more effective for dynamic manipulation due to its `lower encoding latency` (from reduced tokenization) while maintaining `structurally faithful visual representations`, which are critical for timely perception.

### 6.2.3. Continuous Inference (CI)
This study evaluates the impact of `Continuous Inference`.
*   Comparing row `[2]` (CI disabled, `SR 36.11%`) with `[7]` (CI enabled, `SR 47.06%`): Enabling `CI` leads to a substantial increase in `SR` ($+10.95%$) and a reduction in `Task Completion Time` (`9.51s` to `8.53s`).
*   **Conclusion:** `CI` effectively eliminates `inter-chunk waiting`, improving `responsiveness` and overall performance in dynamic tasks.

### 6.2.4. Latent-aware Action Streaming (LAAS)
This study analyzes the contribution of `Latent-aware Action Streaming`.
*   Comparing row `[3]` (LAAS disabled, CI enabled, `SR 39.72%`) with `[7]` (LAAS enabled, CI enabled, `SR 47.06%`): Enabling `LAAS` provides a notable $+7.34%$ increase in `SR` and reduces `Task Completion Time` (`8.84s` to `8.53s`).
*   Comparing row `[1]` (both CI and LAAS disabled, `SR 30.27%`) with `[7]` (both CI and LAAS enabled, `SR 47.06%`): The performance drop is more severe when both are disabled, showing their `complementary roles`.
*   **Conclusion:** Even with `CI` generating actions continuously, `LAAS` is crucial for addressing `temporal misalignment` by discarding outdated actions and prioritizing recent predictions, leading to improved `stability` and `success` in dynamic scenarios.

### 6.2.5. Temporal Visual Context (Appendix D)
This ablation study analyzes how the composition of the `temporal observation window` $\mathbf{O}_t$ affects performance. The default setting is $\mathbf{O}_t = \{ \mathbf{o}_{t-2}, \mathbf{o}_t \}$.

The following are the results from Table III of the original paper:

<table><tr><td>−3</td><td>t-2</td><td>−1</td><td>t</td><td>SR ↑</td><td>PL ↓</td><td>T.Time ↓</td><td>I.Time ↓</td></tr><tr><td>2</td><td>*x</td><td></td><td></td><td>38.22</td><td>2.27</td><td>9.52</td><td>0.225</td></tr><tr><td></td><td></td><td></td><td></td><td>43.39</td><td>2.34</td><td>8.77</td><td>0.226</td></tr><tr><td></td><td></td><td></td><td></td><td>47.06</td><td>2.50</td><td>8.53</td><td>0.226</td></tr><tr><td></td><td></td><td></td><td></td><td>46.89</td><td>2.49</td><td>8.51</td><td>0.226</td></tr><tr><td></td><td>√</td><td>x&gt;××&gt;</td><td></td><td>47.11</td><td>2.49</td><td>8.46</td><td>0.228</td></tr><tr><td></td><td></td><td>L</td><td>2</td><td>47.06</td><td>2.47</td><td>8.53</td><td>0.229</td></tr></table>

*Table III: Ablation on Temporal Visual Context. The temporal observation window is varied by enabling different visual frames at time steps $\{ t - 3 , t - 2 , t - 1 , t \}$ ,while keeping the model architecture, inference frequency, and execution pipeline fixed. Note that SR, PL, T.Time, and I.Time represent the success rate $(\mathrm{in} \%)$ , path length (in meters), task completion time (in seconds), and inference time (in seconds, measured on an NVIDIA RTX A6000 GPU), respectively.*

*   **Single-frame input $\{\mathbf{o}_t\}$:** Results in `38.22% SR`, a clear drop from the `47.06% SR` of the default setting. This confirms that temporal cues are essential for estimating object motion.
*   **$\{\mathbf{o}_{t-1}, \mathbf{o}_t\}$ vs. $\{\mathbf{o}_{t-2}, \mathbf{o}_t\}$:** The setting $\{\mathbf{o}_{t-1}, \mathbf{o}_t\}$ achieves `43.39% SR`, which is lower than the default $\{\mathbf{o}_{t-2}, \mathbf{o}_t\}$ at `47.06% SR`. This suggests that a `larger temporal interval` (`t-2` vs. `t-1`) provides more informative motion cues for `velocity estimation`.
*   **Expanding beyond two frames (e.g., $\{\mathbf{o}_{t-3}, \mathbf{o}_{t-2}, \mathbf{o}_t\}$ or $\{\mathbf{o}_{t-2}, \mathbf{o}_{t-1}, \mathbf{o}_t\}$):** Does not yield further noticeable gains, indicating `diminishing returns` from additional visual redundancy and slightly increasing inference time.
*   **Conclusion:** A `sparse but sufficiently spaced temporal context` (like $\{\mathbf{o}_{t-2}, \mathbf{o}_t\}$) is critical for effective dynamic manipulation without significantly increasing inference frequency.

### 6.2.6. Depth of LLM Backbone (Appendix D)
This study evaluates the effect of truncating the `LLM` backbone by varying the number of `transformer layers` ($l = 8, 16, 24$) and comparing against the full model ($l=32$).

The following are the results from Table IV of the original paper:

<table><tr><td>#Layers</td><td>SR ↑</td><td>PL ↓</td><td>T.Time ↓</td><td>I.Time ↓</td><td>#Param ↓</td></tr><tr><td>8</td><td>44.17</td><td>2.33</td><td>8.92</td><td>0.127</td><td>303</td></tr><tr><td>16</td><td>47.06</td><td>2.50</td><td>8.53</td><td>0.226</td><td>430</td></tr><tr><td>24</td><td>48.44</td><td>2.63</td><td>8.43</td><td>0.317</td><td>558</td></tr><tr><td>32</td><td>42.11</td><td>2.69</td><td>8.39</td><td>0.373</td><td>685</td></tr></table>

*Table IV: Ablation on LLM Depth. Different LLM depths are evaluated by retaining the first $l$ transformer layers. Note that SR, PL, T.Time, I.Time, and #Param denote success rate $(\%)$ , path length (meters), task completion time (seconds), inference time (seconds, measured on an NVIDIA RTX A6000 GPU), and parameter count (in millions), respectively.*

*   **Impact of increasing depth:** Increasing depth from `8 layers` (`SR 44.17%`) to `16 layers` (`SR 47.06%`) generally improves `SR`. Further increasing to `24 layers` yields a slightly higher `SR` (`48.44%`) but with increased `Inference Time (I.Time)`. The full `32-layer` model sees a drop in `SR` (`42.11%`) despite the lowest `T.Time` and highest `I.Time`.
*   **Trade-off:** Aggressively truncating to `8 layers` improves `inference speed` (`0.127s`) but reduces model capacity, leading to a `substantial degradation in success rate`. Increasing depth beyond `16 layers` leads to diminishing returns in `SR` while increasing `I.Time` and `parameter count (#Param)`. The slight gain at `24 layers` might be offset by practical latency concerns.
*   **Conclusion:** The `16-layer` backbone strikes the `optimal balance` between efficiency and robustness (`SR 47.06%`, `I.Time 0.226s`, `430M #Param`). The additional latency from deeper models can be somewhat amortized by `CI` and `LAAS`, but does not necessarily translate to a proportional improvement in success for dynamic tasks.

### 6.2.7. Cross-Model Analysis of CI and LAAS (Appendix D)
This study investigates the generality of `Continuous Inference (CI)` and `Latent-aware Action Streaming (LAAS)` by integrating them into existing `VLA` models without modifying their backbone architectures.

The following are the results from Table V of the original paper:

<table><tr><td>Method</td><td>SR (%) ↑</td><td>PL (m) ↓</td><td>Time (s) ↓</td></tr><tr><td>π0.5† [15]</td><td>15.89</td><td>1.57</td><td>9.95</td></tr><tr><td>SmolVLA† [38]</td><td>25.56</td><td>1.65</td><td>9.77</td></tr><tr><td>DynamicVLA</td><td>47.06</td><td>2.50</td><td>8.53</td></tr></table>

*Table V: Cross-Model Analysis of CI and LAAS. CI and LAAS are integrated into existing VLA models without backbone modification or retraining. Note that SR, PL, and Time represent the success rate (in $\%$) , path length (in meters), and task completion time (in seconds), respectively. † indicates inference-time integration of CI and LAAS.*

*   **SmolVLA†:** Integrating `CI` and `LAAS` into `SmolVLA` improves its `SR` from `12.67%` (Table I) to `25.56%`. This demonstrates that `CI` and `LAAS` effectively enhance `closed-loop responsiveness` when the underlying model (`SmolVLA`) has `moderate inference latency`.
*   **$\pi_{0.5}$†:** Integrating `CI` and `LAAS` into $\pi_{0.5}$ yields only marginal gains, improving `SR` from `11.06%` (Table I) to `15.89%`. This is because $\pi_{0.5}$ has a `substantially larger backbone` (not explicitly stated but implied by "substantially larger backbone" and the small gain) that incurs `very high inference latency`, limiting the effectiveness of overlapping inference and temporally aligned execution.
*   **Conclusion:** `CI` and `LAAS` are `broadly applicable execution mechanisms` that can improve performance in other `VLA` models. However, their practical benefits are `constrained by the underlying inference latency` of the model. Models with inherently high latency will see less dramatic improvements from these execution-level optimizations compared to models designed for efficiency from the ground up, like `DynamicVLA`.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This research effectively tackles the critical challenge of `dynamic object manipulation` for `Vision-Language-Action (VLA)` models. The paper highlights that for such tasks, the primary failure mode is not merely perceptual ambiguity, but rather the `temporal misalignment` between the robot's observations and the execution of its actions, a factor often overlooked in `static manipulation` research.

`DynamicVLA` addresses this through three key innovations:
1.  **A compact `0.4B` parameter `VLA` backbone:** This architecture is designed for `high-frequency reasoning` by employing an efficient `convolutional vision encoder (FastViT)` and a truncated language model, minimizing `inference latency`.
2.  **`Continuous Inference (CI)`:** A `pipelined execution scheme` that `overlaps reasoning and execution`, eliminating `inter-chunk waiting` and ensuring timely adaptation to object motion.
3.  **`Latent-aware Action Streaming (LAAS)`:** A `latency-aware execution mechanism` that `discards outdated actions` and `prioritizes the most recent predictions`, enforcing `temporally aligned action execution`.

    Furthermore, to overcome the scarcity of data for dynamic manipulation, the paper introduces the `Dynamic Object Manipulation (DOM)` benchmark. This benchmark features automated data collection pipelines that efficiently gather `200K synthetic episodes` and `2K real-world episodes` (without teleoperation) across multiple robot embodiments.

Extensive evaluations demonstrate that these integrated elements significantly reduce the `perception-execution gap`, leading to substantially more responsive and successful behavior in dynamic manipulation tasks compared to conventional `VLA` models.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations of the current study and propose promising directions for future research:

*   **More Efficient VLA Architectures:** `DynamicVLA` emphasizes latency-aware design, but a fundamental trade-off exists between `multimodal understanding` and `responsiveness`. Future work needs to explore architectures and inference schemes that can `preserve deep understanding` under even stricter latency budgets, as dynamic tasks tightly couple perception, reasoning, and execution.
*   **Beyond Short-horizon Dynamics:** The current formulation primarily focuses on `short- to medium-horizon reactive interaction`. This effectively exposes latency-induced failures but does not extend to `longer-horizon dynamic behaviors`. Future work should integrate `planning`, `memory`, and `task decomposition` for `multi-stage tasks` with persistent object motion, all while remaining compatible with `language conditioning` and `real-time execution constraints`.
*   **Beyond Rigid-Body Dynamics:** The `DOM` data pipeline currently assumes `rigid-body state estimation`. However, many real-world dynamic tasks involve `non-rigid` or `fluid dynamics`, where objects have continuously evolving states that are difficult to represent and model accurately in both simulation and the real world. Extending `VLA` models and data pipelines to handle such complex dynamics remains an open challenge.

## 7.3. Personal Insights & Critique
`DynamicVLA` presents a compelling and timely solution to a critical bottleneck in robotic manipulation: the handling of dynamic objects. Its core insight—that latency, not just perceptual accuracy, is the dominant failure mode in dynamic scenarios—is profoundly important and often overlooked in the pursuit of larger, more capable `VLM`s.

**Strengths:**
*   **Holistic Approach:** The paper doesn't just propose a model; it addresses the entire ecosystem: model architecture, inference strategy, action execution, and data generation. This comprehensive approach is crucial for real-world impact.
*   **Pragmatic Design:** The choice of a compact `0.4B` `VLA` and a `convolutional vision encoder` (`FastViT`) demonstrates a deep understanding of hardware constraints and real-time requirements, prioritizing efficiency without sacrificing core capabilities.
*   **Innovative Execution Mechanisms (`CI` and `LAAS`):** These are the standout contributions. `Continuous Inference` and `Latent-aware Action Streaming` are elegant solutions that directly tackle the `perception-execution gap` at the control level, allowing the robot to adapt proactively rather than reactively to stale information. The ablation studies convincingly support their efficacy.
*   **Foundational Benchmark (`DOM`):** The creation of a large-scale, automated `DOM` benchmark is a monumental contribution. The automated real-world data collection, bypassing teleoperation for dynamic tasks, is particularly ingenious and addresses a long-standing challenge in robotics data collection. This benchmark will likely serve as a crucial resource for future research in dynamic manipulation.
*   **Rigorous Evaluation:** The extensive evaluation across simulation and multiple real-world embodiments, covering interaction, perception, and generalization, provides strong evidence for the effectiveness of `DynamicVLA`.

**Potential Issues/Critique:**
*   **Path Length Metric:** While `DynamicVLA` achieves significantly lower task completion times and higher success rates, its `Path Length` is higher than baselines. The paper mentions this could be due to `more intricate movements` needed for adaptation. However, this might also indicate less optimal or less "human-like" motion planning. Future work could explore if `Path Length` can be optimized without sacrificing reactivity.
*   **Disturbance Robustness:** Even `DynamicVLA` struggles with `Disturbance Robustness` in simulation. This suggests that while it handles predictable and somewhat unpredictable motion, truly unexpected external perturbations remain a tough nut to crack. This is a common challenge in robotics, but `DynamicVLA`'s current architecture might need more explicit mechanisms for robust replanning or force control in the face of strong, unexpected forces.
*   **Generalizability of `CI` and `LAAS`:** While the cross-model analysis shows `CI` and `LAAS` are beneficial for `SmolVLA`, they only yield marginal gains for $\pi_{0.5}$ due to its larger backbone and higher latency. This reinforces the idea that these execution-level optimizations are most impactful when paired with an inherently efficient backbone. It means `DynamicVLA`'s success is deeply intertwined with its architectural choices, rather than `CI` and `LAAS` being universal "fixes" for any `VLA`.
*   **"Real-world simulator" Details:** While innovative, more technical details on the latency, accuracy, and robustness of the "real-world simulator" (EfficientTAM, geometric triangulation, motion fitting) would be valuable. The sim-to-real gap, even with this system, likely still exists due to sensor noise and estimation errors.

**Applicability and Future Value:**
The methods proposed in `DynamicVLA` hold immense potential for applications beyond general object manipulation. Concepts like `Continuous Inference` and `Latent-aware Action Streaming` are broadly applicable to any robotic system requiring `real-time, low-latency control` in dynamic environments, such as drone navigation, autonomous driving, or human-robot collaboration where precise timing is critical. The `DOM` benchmark will undoubtedly accelerate research in dynamic robotic tasks, providing a standardized platform that was previously lacking. The paper's emphasis on balancing model capacity with inference efficiency will guide the development of future `VLA` models for real-world deployment, where computational resources and latency are often primary constraints.