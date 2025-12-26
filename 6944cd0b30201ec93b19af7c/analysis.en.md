# 1. Bibliographic Information

## 1.1. Title
A Comprehensive Survey on World Models for Embodied AI

## 1.2. Authors
The authors of this paper are Xinqing Li, Xin He, Le Zhang (Member, IEEE), Min Wu (Senior Member, IEEE), Xiaoli Li (Fellow, IEEE), and Yun Liu.

The authors' affiliations are primarily with Nanyang Technological University (NTU), Singapore, a globally renowned institution for engineering and technology. The inclusion of IEEE Members, Senior Members, and a Fellow (Xiaoli Li) indicates a team with significant experience and standing within the research community, lending credibility to this comprehensive survey.

## 1.3. Journal/Conference
This paper is a preprint available on arXiv. The provided metadata indicates a future publication date (October 19, 2025), which is likely a placeholder. The arXiv identifier (`2510.16732`) also suggests a submission in October 2025. Given that the paper cites numerous works from late 2024 and early 2025 conferences (e.g., AAAI'25, CVPR'25, ICLR'25), it represents a state-of-the-art overview of the field as of that time. An arXiv preprint is not peer-reviewed, but it serves as a way to rapidly disseminate important research findings to the academic community.

## 1.4. Publication Year
The metadata lists 2025, but as a preprint, the content reflects the state of research in late 2024 and early 2025.

## 1.5. Abstract
The abstract introduces world models as internal simulators that are critical for embodied AI agents, enabling them to perceive, act, and predict the consequences of their actions. The paper's primary contribution is a unified framework and a three-axis taxonomy for classifying world models: (1) **Functionality** (Decision-Coupled vs. General-Purpose), (2) **Temporal Modeling** (Sequential Simulation vs. Global Difference Prediction), and (3) **Spatial Representation** (Global Latent Vector, Token Feature Sequence, Spatial Latent Grid, Decomposed Rendering Representation). The survey also systematizes datasets and evaluation metrics across robotics, autonomous driving, and general video domains. It provides a quantitative comparison of state-of-the-art models and identifies key open challenges, such as the need for unified datasets, physically consistent evaluation metrics, balancing performance with computational efficiency, and achieving long-horizon temporal consistency. The authors also maintain a curated bibliography for ongoing updates.

## 1.6. Original Source Link
- **Original Source (arXiv):** https://arxiv.org/abs/2510.16732
- **PDF Link:** https://arxiv.org/pdf/2510.16732v2.pdf
- **Status:** This is a preprint and has not yet undergone formal peer review.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem addressed by this survey is the lack of a unified structure and consistent terminology in the rapidly expanding field of world models for embodied AI. Embodied AI agents—such as robots or autonomous vehicles—must interact with and understand the physical world. To do this effectively, they need an "internal model" of how the world works, allowing them to simulate the future and plan their actions. These are known as **world models**.

While the concept has roots in model-based reinforcement learning, recent advances in generative AI have led to an explosion of different approaches. Researchers in robotics, autonomous driving, and video generation have all developed world models, but often use different architectures, terminology, and evaluation methods. This fragmentation makes it difficult to compare methods, understand the trade-offs between different design choices, and identify overarching research trends.

The paper's innovative entry point is to cut through this complexity by proposing a clear, comprehensive taxonomy. Instead of focusing on a single application (like driving) or a single function (like prediction), the authors identify three fundamental design axes that are common to all world models: their functional purpose, how they model time, and how they represent space.

## 2.2. Main Contributions / Findings
This survey makes several key contributions to the field:

1.  **A Unified Three-Axis Taxonomy:** The central contribution is a novel taxonomy for categorizing world models. This provides a clear and structured way to understand the vast landscape of existing research. The three axes are:
    *   **Functionality:** Distinguishes between `Decision-Coupled` models (optimized for a specific task like robot control) and `General-Purpose` models (trained as task-agnostic simulators).
    *   **Temporal Modeling:** Differentiates between `Sequential Simulation and Inference` (autoregressive, step-by-step prediction) and `Global Difference Prediction` (parallel, one-shot prediction of a future state).
    *   **Spatial Representation:** Classifies how models encode the world's state, ranging from abstract to explicit geometric representations: `Global Latent Vector`, `Token Feature Sequence`, `Spatial Latent Grid`, and `Decomposed Rendering Representation`.

2.  **Systematic Review of the Field:** The paper applies this taxonomy to systematically review and organize a large body of representative works in robotics and autonomous driving. This creates a "knowledge map" that clarifies the relationships between different models and research directions.

3.  **Consolidation of Resources and Metrics:** It compiles and organizes essential data resources (simulators, benchmarks, datasets) and evaluation metrics. By explaining metrics for pixel quality, state-level understanding, and task performance, it provides a standardized foundation for evaluating and comparing world models.

4.  **Quantitative Comparison and Identification of Challenges:** The survey presents quantitative performance comparisons of state-of-the-art models on key benchmarks. Based on this analysis, it distills critical open challenges, including the need for better datasets, physics-aware evaluation, improved computational efficiency for real-time control, and mitigating error accumulation in long-horizon predictions.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To fully grasp this paper, one must understand several core concepts from AI and machine learning.

### 3.1.1. Embodied AI
**Embodied AI** refers to artificial intelligence systems that exist within a physical or simulated environment and can interact with it through sensors (perception) and actuators (action). Unlike disembodied AI (e.g., a chatbot or a language model processing text), an embodied agent's understanding is grounded in its physical interactions. This means it must learn not just to recognize patterns but to understand cause and effect—how its actions change the state of the world. Examples include autonomous cars, robot manipulators, and navigation agents in virtual worlds.

### 3.1.2. World Model
A **world model** is an internal, learned representation of an environment's dynamics. It functions as a "mental simulator" for an AI agent. By learning the rules of how the world evolves, a world model allows the agent to:
*   **Predict the Future:** Given the current state and a potential sequence of actions, it can "imagine" or "roll out" what the future states will look like.
*   **Plan and Make Decisions:** The agent can test out different action sequences in its imagination to find the one that leads to the best outcome, without costly or dangerous real-world trial and error.
*   **Understand Counterfactuals:** It can reason about "what would have happened if" a different action had been taken.
*   **Improve Perception:** It can use its understanding of dynamics to fill in missing information, such as inferring the state of occluded objects.

### 3.1.3. Partially Observable Markov Decision Process (POMDP)
A **POMDP** is a mathematical framework for modeling decision-making in situations where the agent cannot directly observe the true state of the world. This is a perfect fit for embodied AI, where sensors like cameras only provide a partial, noisy view of the environment.

A POMDP is defined by:
*   $S$: A set of true world states (e.g., the exact 3D positions and velocities of all objects). These are **hidden** from the agent.
*   $A$: A set of actions the agent can take.
*   $T(s' | s, a)$: The transition function, which gives the probability of moving to state $s'$ from state $s$ after taking action $a$.
*   `R(s, a)`: The reward function, which gives the immediate reward for taking action $a$ in state $s$.
*   $\Omega$: A set of observations the agent can receive.
*   $O(o | s', a)$: The observation function, which gives the probability of receiving observation $o$ after taking action $a$ and landing in state $s'$.
*   $\gamma$: A discount factor for future rewards.

    The agent's goal is to choose actions that maximize its expected cumulative reward, based only on its history of observations and actions. World models are a way to solve POMDPs by learning to infer a belief about the hidden state from observations and then using a learned transition model to predict future states.

### 3.1.4. Variational Inference and the Evidence Lower Bound (ELBO)
In the context of world models, we want to learn the parameters of our model (e.g., a neural network) from data. A principled way to do this is to maximize the probability of the observations we've seen, which is called the **log-likelihood**, $\log p(o_{1:T})$. However, this is often mathematically intractable to compute directly because it requires integrating over all possible latent state sequences.

**Variational Inference (VI)** is a technique to approximate this intractable posterior distribution. Instead of computing the true posterior $p(z | o)$, we define a simpler, tractable family of distributions $q_{\phi}(z | o)$ (parameterized by $\phi$) and try to make it as close as possible to the true posterior.

The **Evidence Lower Bound (ELBO)** is the objective function we maximize to achieve this. It is a lower bound on the log-likelihood:
\$
\log p(o) \geq \mathbb{E}_{q_{\phi}(z|o)} \left[ \log \frac{p_{\theta}(o, z)}{q_{\phi}(z|o)} \right] =: \mathcal{L}(\theta, \phi)
\$
Maximizing the ELBO has two effects: it pushes the log-likelihood $\log p(o)$ up, and it minimizes the KL divergence (a measure of difference) between our approximation $q_{\phi}$ and the true posterior $p_{\theta}(z|o)$. As the paper shows, this ELBO can be decomposed into a **reconstruction term** (how well the model can reconstruct observations from latent states) and a **regularization term** (how well the learned dynamics match the states inferred from data).

## 3.2. Previous Works
The paper categorizes previous surveys on world models into two main types:

1.  **Function-Oriented Surveys:** These surveys organize works based on the core functions of world models. For example, Ding et al. (2024) used the categories of "understanding" and "prediction." This approach focuses on *what* the models do.
2.  **Application-Driven Surveys:** These surveys focus on a specific domain. For example, Guan et al. (2024) and Feng et al. (2025) reviewed world models specifically for autonomous driving. This approach focuses on *where* the models are used.

    This paper differentiates itself by proposing a more fundamental taxonomy based on the core design choices of the models themselves, which is applicable across different functions and applications.

The survey builds upon a rich history of research, with some seminal works being:
*   **Ha and Schmidhuber (2018):** This paper is credited with popularizing the term "world model." It showed that an agent could learn a compressed latent representation of its environment and a recurrent neural network (RNN) to model temporal dynamics. This learned world model was then used to train a policy entirely in "imagination," which was surprisingly effective.
*   **The Dreamer Series (PlaNet, Dreamer, DreamerV2, DreamerV3):** This line of work, led by Danijar Hafner, developed the **Recurrent State-Space Model (RSSM)**, a sophisticated architecture for learning world models from images. The RSSM is a cornerstone of many modern world models. It combines a deterministic component (like an RNN hidden state) with a stochastic latent variable at each timestep. This allows it to model both predictable dynamics and inherent uncertainty in the environment. The learning objective is derived from the ELBO, balancing reconstruction of the input image with consistency between the predicted and observed latent states.

## 3.3. Technological Evolution
The field of world models has evolved significantly:
*   **Early Stages (Model-Based RL):** The initial focus was on improving the sample efficiency of reinforcement learning. Small-scale latent state-transition models were learned to allow for planning, reducing the need for extensive real-world interaction.
*   **The Rise of Deep Generative Models (Dreamer-era):** With the success of deep learning, models like `PlaNet` and `Dreamer` demonstrated that world models could be learned directly from high-dimensional pixel inputs, enabling control in complex visual environments like the DeepMind Control Suite. The core architecture was often an RNN-based state-space model.
*   **The Transformer and Large-Scale Pretraining Era:** More recently, architectures like Transformers and Diffusion Models have been adopted. These models, pretrained on massive datasets of videos and multimodal data (e.g., `Sora`, `V-JEPA 2`), have shifted the focus from purely policy-learning aids to **general-purpose simulators**. They can generate high-fidelity, long-horizon video predictions and are increasingly used for a wider range of downstream tasks beyond just control. This expansion has led to the diversification of architectures and the fragmentation that this survey aims to address.

## 3.4. Differentiation Analysis
Compared to previous surveys, this paper's core innovation lies in its **unifying and multi-dimensional taxonomy**.

*   **Breadth over Depth in a Single Area:** While application-specific surveys provide deep dives into autonomous driving, this paper's framework is general enough to connect insights from robotics, driving, and general video modeling.
*   **Architectural Focus over Functional Focus:** Instead of just classifying by "understanding" vs. "prediction," the paper's axes of `Temporal Modeling` and `Spatial Representation` delve into the *how*—the architectural choices that enable these functions. This provides a more actionable guide for researchers designing new models.
*   **Integration and Systematization:** A major contribution is not just the taxonomy itself but its application. By using it to systematize datasets, metrics, and a large corpus of recent papers (including those from 2024 and 2025), it provides a holistic and up-to-date "state of the union" for the entire field.

# 4. Methodology

The core methodology of this survey is twofold: first, to establish a formal mathematical foundation for world models, and second, to propose a comprehensive three-axis taxonomy to organize the field.

## 4.1. Principles
The guiding principle is that any world model, regardless of its specific application or architecture, must fundamentally address the challenge of modeling an environment's dynamics. This involves representing the state of the world (spatially) and predicting how that state evolves over time (temporally). The paper formalizes this process using the language of probabilistic graphical models and variational inference, providing a common mathematical ground for all world models. It then uses this foundation to derive a taxonomy based on the key design choices that different models make to solve this core problem.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Mathematical Formulation of World Models

The paper formalizes the problem of an agent interacting with its environment as a **Partially Observable Markov Decision Process (POMDP)**. This section provides a step-by-step breakdown of their formulation.

**Step 1: Defining the Interaction**
At each discrete timestep $t$, the agent receives an observation $o_t$ (e.g., a camera image) from the environment. The true underlying state of the world, $s_t$, is hidden. The agent then takes an action $a_t$. Because the history of observations can be long and complex, world models aim to summarize all relevant past information into a compact **latent state**, denoted as $z_t$.

**Step 2: The Generative Process and Key Components**
The paper defines a generative model that describes how observations are produced. This model, parameterized by $\theta$, consists of three key probabilistic components:

1.  **Dynamics Prior:** This model predicts the next latent state $z_t$ based on the previous latent state $z_{t-1}$ and the action taken $a_{t-1}$. It represents the world model's internal simulation of physics or dynamics.
    \$
    p_{\theta}(z_t | z_{t-1}, a_{t-1})
    \$
2.  **Reconstruction (or Observation Model):** This model generates or reconstructs the observation $o_t$ from the current latent state $z_t$. It connects the abstract latent state back to the sensory data.
    \$
    p_{\theta}(o_t | z_t)
    \$
3.  **Filtered Posterior (or Inference Model):** This is an inference model, parameterized by $\phi$, that refines the prediction of the latent state $z_t$ by incorporating the new observation $o_t$. It represents how the agent updates its belief about the world after seeing new evidence.
    \$
    q_{\phi}(z_t | z_{t-1}, a_{t-1}, o_t)
    \$

Under this structure, the joint probability of a sequence of observations and latent states over a horizon $T$ is given by:
\$
p_{\theta}(o_{1:T}, z_{0:T} | a_{0:T-1}) = p_{\theta}(z_0) \prod_{t=1}^{T} p_{\theta}(z_t | z_{t-1}, a_{t-1}) p_{\theta}(o_t | z_t)
\$
Here, $p_{\theta}(z_0)$ is the initial state distribution, and the product shows how the state evolves and generates observations at each step.

**Step 3: The Learning Objective (ELBO)**
The goal is to train the model parameters ($\theta$ and $\phi$) by maximizing the log-likelihood of the observed data, $\log p_{\theta}(o_{1:T} | a_{0:T-1})$. This is intractable because it requires integrating over all possible latent state sequences $z_{0:T}$.

To solve this, the paper uses variational inference, introducing an approximate posterior distribution $q_{\phi}$ that is structured to be tractable:
\$
q_{\phi}(z_{0:T} | o_{1:T}, a_{0:T-1}) = q_{\phi}(z_0 | o_1) \prod_{t=1}^{T} q_{\phi}(z_t | z_{t-1}, a_{t-1}, o_t)
\$

We then maximize the **Evidence Lower Bound (ELBO)**, $\mathcal{L}(\theta, \phi)$, which is a tractable lower bound on the log-likelihood:
\$
\begin{aligned}
\log p_{\theta}(o_{1:T} | a_{0:T-1}) &= \log \int p_{\theta}(o_{1:T}, z_{0:T} | a_{0:T-1}) dz_{0:T} \\
&\geq \mathbb{E}_{q_{\phi}} \left[ \log \frac{p_{\theta}(o_{1:T}, z_{0:T} | a_{0:T-1})}{q_{\phi}(z_{0:T} | o_{1:T}, a_{0:T-1})} \right] =: \mathcal{L}(\theta, \phi)
\end{aligned}
\$
By substituting the factorized forms of $p_{\theta}$ and $q_{\phi}$, the ELBO can be decomposed into two main parts summed over all timesteps:

\$
\mathcal{L}(\theta, \phi) = \sum_{t=1}^{T} \mathbb{E}_{q_{\phi}(z_t)} \bigl[ \log p_{\theta}(o_t | z_t) \bigr] - D_{\mathrm{KL}} \bigl( q_{\phi}(z_{0:T} | o_{1:T}, a_{0:T-1}) \parallel p_{\theta}(z_{0:T} | a_{0:T-1}) \bigr)
\$

*   **First Term (Reconstruction Loss):** $\log p_{\theta}(o_t | z_t)$ is the log-likelihood of reconstructing the observation $o_t$ from the latent state $z_t$. Maximizing this term forces the latent state to contain all the information necessary to generate the sensory input.
*   **Second Term (KL Regularization):** The Kullback-Leibler (KL) divergence term $D_{\mathrm{KL}}(\cdot \parallel \cdot)$ measures the "distance" between the posterior distribution $q_{\phi}$ (inferred using the observation) and the dynamics prior distribution $p_{\theta}$ (predicted without the observation). Minimizing this term encourages the dynamics model to make predictions that are consistent with what is actually observed. This forces the model to learn the true dynamics of the environment.

    This **reconstruction-regularization** framework is the theoretical backbone for training most modern world models, such as the `Dreamer` series.

### 4.2.2. The Three-Axis Taxonomy

The paper's main contribution is a taxonomy for classifying world models along three orthogonal axes. The following diagram from the paper provides a visual overview.

![该图像是一个示意图，展示了关于世界模型在体感人工智能中的框架。图中包括了核心概念、决策耦合、空间表示和时间建模等模块。每个模块下列出了相关的方法和代表性模型，以及它们在不同任务中的应用，如视频生成和全局预测。](images/1.jpg)

**Axis 1: Functionality**
This axis describes the ultimate purpose of the world model.

*   **Decision-Coupled:** These models are trained with a specific downstream task in mind, typically decision-making or control. The latent space and dynamics are optimized to be useful for a policy or planner. For example, the world model might only learn to represent dynamics relevant to picking up an object, while ignoring irrelevant details like wall textures. **Example:** `Dreamer` trains a world model and a policy together to maximize task rewards.
*   **General-Purpose:** These models are task-agnostic. They are trained to be high-fidelity simulators of the environment, focusing on accurately predicting future sensory data (like video frames) without being tied to a specific task. The goal is to create a foundational model of the world that can later be adapted for various downstream tasks. **Example:** `Sora` is trained on a massive dataset of videos to generate realistic future frames, without an explicit control task.

**Axis 2: Temporal Modeling**
This axis describes how the model predicts the evolution of the world over time.

*   **Sequential Simulation and Inference:** These models operate autoregressively, predicting the future one step at a time. The state at time $t$ is used to predict the state at $t+1$, which is then used to predict $t+2$, and so on. This is analogous to how traditional physics simulators work.
    *   **Advantage:** Computationally efficient per step and naturally suited for closed-loop control where an agent's action at each step influences the next.
    *   **Disadvantage:** Prone to **error accumulation**, where small prediction errors at early steps compound over long horizons, leading to unrealistic or divergent rollouts.
    *   **Example:** `Dreamer` uses an RSSM (which is recurrent) to predict latent states sequentially.
*   **Global Difference Prediction:** These models predict an entire future sequence or a distant future state in a single, parallel pass. Instead of stepping through time, they might take the current state and directly generate the state 5 seconds into the future.
    *   **Advantage:** Less susceptible to compounding errors and often computationally faster for generating long sequences.
    *   **Disadvantage:** Can struggle with temporal coherence and fine-grained, step-by-step interactivity needed for closed-loop control.
    *   **Example:** `V-JEPA` predicts features of future video chunks in parallel from a masked context.

**Axis 3: Spatial Representation**
This axis describes how the model encodes the state of the world at a single point in time.

*   **Global Latent Vector:** The entire world state is compressed into a single, compact, flat vector of numbers (e.g., a 1024-dimensional vector).
    *   **Advantage:** Highly efficient for computation, making it suitable for real-time control on resource-constrained hardware.
    *   **Disadvantage:** Lacks explicit spatial structure. It's difficult to represent the geometric relationships between objects, which can limit performance on tasks requiring spatial reasoning.
    *   **Example:** The original `World Models` paper by Ha & Schmidhuber used this representation.
*   **Token Feature Sequence:** The world state is represented as a sequence of discrete tokens or feature vectors, similar to how text is represented in a Large Language Model (LLM). These tokens can represent objects, patches of an image, or different modalities.
    *   **Advantage:** Leverages the power of Transformer architectures to model complex dependencies between tokens. It is highly flexible and can naturally handle multimodal inputs (vision, language, action).
    *   **Example:** `Genie` represents interactive environments using discrete spatiotemporal tokens.
*   **Spatial Latent Grid:** The representation incorporates an explicit spatial inductive bias by arranging features on a geometric grid. This can be a 2D Bird's-Eye View (BEV) grid common in autonomous driving, or a 3D voxel grid.
    *   **Advantage:** Preserves spatial locality, making it easy to use convolutions or local attention. It is well-suited for tasks that require geometric understanding, like navigation and collision avoidance.
    *   **Example:** `OccWorld` predicts future 3D occupancy grids for autonomous driving.
*   **Decomposed Rendering Representation:** The scene is represented by a set of explicit, renderable 3D primitives. This includes methods like **Neural Radiance Fields (NeRF)**, which use a neural network to represent a continuous volumetric scene, and **3D Gaussian Splatting (3DGS)**, which represents the scene as a collection of 3D Gaussians. The world model learns to predict how these primitives move and change over time.
    *   **Advantage:** Provides extremely high-fidelity, view-consistent rendering. It naturally handles object permanence and 3D geometry.
    *   **Disadvantage:** Computationally very expensive and can be difficult to scale to complex, dynamic scenes.
    *   **Example:** `ManiGaussian` uses 3DGS to model dynamic scenes for robotic manipulation.

# 5. Experimental Setup

This being a survey paper, the "Experimental Setup" section consolidates the datasets, metrics, and baseline models used across the literature to evaluate world models.

## 5.1. Datasets
The paper organizes data resources into four categories, highlighting the most influential ones used for training and benchmarking world models.

### 5.1.1. Simulation Platforms
These provide controllable, scalable virtual environments.
*   `MuJoCo`: A fast and popular physics engine for robotics, used for continuous control tasks.
*   `CARLA`: An open-source simulator for autonomous driving research, with realistic sensors and urban environments.
*   `Habitat`: A high-performance simulator for embodied AI, focusing on indoor navigation in photorealistic 3D environments.
*   `NVIDIA Isaac (Sim, Gym, Lab)`: A GPU-accelerated robotics simulation platform offering photorealistic rendering and large-scale reinforcement learning capabilities.

### 5.1.2. Interactive Benchmarks
These are standardized task suites for reproducible evaluation.
*   `DeepMind Control (DMC) Suite`: A set of continuous control tasks (e.g., 'Cheetah Run', 'Walker Walk') based on `MuJoCo`, widely used to benchmark world models learned from pixels.
*   `Atari`: A classic suite of 2D pixel-based video games for evaluating reinforcement learning agents.
*   `RLBench`: A challenging benchmark with 100 diverse robotic manipulation tasks.
*   `nuPlan`: A large-scale, real-world planning benchmark for autonomous driving, featuring closed-loop simulation.
*   `LIBERO`: A benchmark for lifelong robotic manipulation, designed to test continual learning.

### 5.1.3. Offline Datasets
These are large-scale, pre-collected datasets used for training and offline evaluation.
*   `nuScenes`: A large-scale multimodal dataset for autonomous driving, with 360-degree sensor data (cameras, LiDAR, radar) and detailed 3D annotations.
*   `Waymo Open Dataset`: Another large-scale autonomous driving dataset with high-resolution sensor data.
*   `Open X-Embodiment (OXE)`: A massive corpus of robot learning data, aggregating over a million trajectories from 22 different robot embodiments, designed to train generalist robot policies.
*   `OpenDV`: A very large video-text dataset for autonomous driving, collected from public sources to support pretraining of driving world models.
*   `VideoMix22M`: A large-scale video dataset used to pretrain the `V-JEPA 2` model, containing over 22 million samples from various sources.
*   An example of a data sample from a robotics dataset like `RT-1` would be a trajectory containing:
    *   **Language Instruction:** "pick up the apple"
    *   **Image Observations:** A sequence of camera images from the robot's perspective.
    *   **Actions:** A sequence of recorded motor commands (e.g., discretized movements of the robot arm and base).

### 5.1.4. Real-world Robot Platforms
Physical robots used for validation in the real world.
*   `Franka Emika Panda`: A popular 7-DoF collaborative robot arm.
*   `Unitree Go1 / G1`: Widely used quadrupedal and humanoid robots for locomotion and manipulation research.

## 5.2. Evaluation Metrics
The paper groups metrics into three levels of abstraction.

### 5.2.1. Pixel Generation Quality
These metrics assess the visual fidelity of generated videos.

*   **Fréchet Inception Distance (FID)**
    *   **Conceptual Definition:** Measures the similarity between the distribution of real images and generated images. It captures both fidelity (realism) and diversity (variety). A lower FID is better.
    *   **Mathematical Formula:**
        \$
        \mathrm{FID}(x, y) = \| \boldsymbol{\mu_x} - \boldsymbol{\mu_y} \|_2^2 + \mathrm{Tr}\left( \boldsymbol{\Sigma_x} + \boldsymbol{\Sigma_y} - 2(\boldsymbol{\Sigma_x}\boldsymbol{\Sigma_y})^{1/2} \right)
        \$
    *   **Symbol Explanation:**
        *   $\boldsymbol{\mu_x}, \boldsymbol{\mu_y}$: Mean vectors of the Inception-v3 feature embeddings for real ($x$) and generated ($y$) images.
        *   $\boldsymbol{\Sigma_x}, \boldsymbol{\Sigma_y}$: Covariance matrices of the feature embeddings.
        *   $\mathrm{Tr}(\cdot)$: The trace of a matrix (sum of diagonal elements).

*   **Fréchet Video Distance (FVD)**
    *   **Conceptual Definition:** An extension of FID to videos. It evaluates both per-frame image quality and temporal consistency (smoothness and realism of motion). A lower FVD is better.
    *   **Mathematical Formula:** The formula is identical to FID, but the features are extracted from a video-classification network (like I3D) instead of an image network.

*   **Structural Similarity Index Measure (SSIM)**
    *   **Conceptual Definition:** Measures the perceptual similarity between two images based on their luminance, contrast, and structure. A score of 1 indicates perfect similarity.
    *   **Mathematical Formula:**
        \$
        \mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
        \$
    *   **Symbol Explanation:**
        *   $\mu_x, \mu_y$: Means of image patches $x$ and $y$.
        *   $\sigma_x^2, \sigma_y^2$: Variances of patches $x$ and $y$.
        *   $\sigma_{xy}$: Covariance of $x$ and $y$.
        *   $C_1, C_2$: Small constants to stabilize the division.

*   **Peak Signal-to-Noise Ratio (PSNR)**
    *   **Conceptual Definition:** Measures the ratio between the maximum possible power of a signal and the power of corrupting noise. In images, it measures pixel-wise reconstruction quality. Higher PSNR is better.
    *   **Mathematical Formula:**
        \$
        \mathrm{PSNR}(x, y) = 10 \cdot \log_{10}\left(\frac{\mathrm{MAX}^2}{\mathrm{MSE}}\right)
        \$
    *   **Symbol Explanation:**
        *   $\mathrm{MAX}$: Maximum possible pixel value (e.g., 255).
        *   $\mathrm{MSE} = \frac{1}{N} \sum_{i=1}^{N}(x_i - y_i)^2$: Mean Squared Error between the two images.

*   **Learned Perceptual Image Patch Similarity (LPIPS)**
    *   **Conceptual Definition:** Measures the perceptual distance between two images by comparing their deep feature representations from a pretrained neural network. It aligns better with human judgment than PSNR or SSIM. Lower LPIPS is better.
    *   **Mathematical Formula:**
        \$
        \mathrm{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot (\hat{f}_{h,w,x}^l - \hat{f}_{h,w,y}^l) \right\|_2^2
        \$
    *   **Symbol Explanation:**
        *   $\hat{f}^l$: Normalized activations from layer $l$ of a deep network for images $x$ and $y$.
        *   $w_l$: Channel-wise weights to scale the importance of each channel.
        *   $H_l, W_l$: Height and width of the feature map at layer $l$.

### 5.2.2. State-level Understanding
These metrics evaluate the model's understanding of the scene's structure and semantics.

*   **mean Intersection over Union (mIoU)**
    *   **Conceptual Definition:** A standard metric for semantic segmentation. It measures the overlap between the predicted segmentation mask and the ground truth mask, averaged over all classes. Higher mIoU is better.
    *   **Mathematical Formula:** For a single class, $\mathrm{IoU} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}$. The final metric is the mean over all classes:
        \$
        \mathrm{mIoU} = \frac{1}{|C|} \sum_{c \in C} \mathrm{IoU}_c
        \$
    *   **Symbol Explanation:**
        *   $\mathrm{TP}, \mathrm{FP}, \mathrm{FN}$: True Positives, False Positives, and False Negatives.
        *   $C$: The set of all classes.

*   **Chamfer Distance (CD)**
    *   **Conceptual Definition:** Measures the geometric dissimilarity between two point clouds. It calculates the average closest point distance between them. Lower CD is better.
    *   **Mathematical Formula:**
        \$
        \mathrm{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} \|x - y\|_2^2 + \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|_2^2
        \$
    *   **Symbol Explanation:**
        *   $S_1, S_2$: The two point sets (e.g., predicted and ground truth point clouds).

### 5.2.3. Task Performance
These metrics evaluate how well the world model supports an agent in achieving its goals.

*   **Success Rate (SR):** The percentage of trials where the agent successfully completes its task (e.g., reaches a goal, manipulates an object correctly).
*   **Sample Efficiency (SE):** The amount of data or interaction steps required to reach a certain level of performance. Higher sample efficiency is better.
*   **Reward:** In reinforcement learning, the cumulative discounted or average reward obtained by the agent over an episode.
*   **Collision Rate:** The percentage of trials in which the agent collides with an obstacle. This is a critical safety metric in navigation and autonomous driving.

## 5.3. Baselines
The "baselines" in this survey are the various state-of-the-art world models that are compared against each other in the results section. These include models like `Dreamer`, `GenAD`, `OccWorld`, `ManiGaussian`, `UniAD`, and many others mentioned throughout the paper. They are representative because they cover the different categories of the proposed taxonomy and are often top performers on established benchmarks.

# 6. Results & Analysis

The paper provides a quantitative comparison of state-of-the-art models across several key tasks. This analysis reveals important trends and trade-offs in world model design.

## 6.1. Core Results Analysis

### 6.1.1. Pixel Generation on nuScenes (Table IV)
This task evaluates a model's ability to generate realistic driving videos. The key metrics are `FID` (image fidelity) and `FVD` (video consistency).
*   **Observation:** There has been dramatic progress in generation quality. Models like `DrivePhysica` (FID: 4.0) and `MiLA` (FVD: 14.9) achieve state-of-the-art results. `MiLA`'s extremely low FVD suggests it excels at temporal coherence.
*   **Analysis:** High-resolution generation (e.g., `Vista` at 576x1024) is now feasible and achieves excellent FID scores (6.9). The use of diffusion-based backbones and advanced techniques like physics-informed modeling (`DrivePhysica`) and coarse-to-fine generation (`MiLA`) are driving these improvements.

### 6.1.2. Scene Understanding: 4D Occupancy Forecasting on Occ3D-nuScenes (Table V)
This task assesses a model's ability to predict the 3D geometry and occupancy of a scene over time. The primary metric is `mIoU`.
*   **Observation:** The results show a clear hierarchy. Models using ground-truth occupancy (`Occ`) as input significantly outperform camera-only models. Furthermore, providing the ground-truth ego-vehicle trajectory (`GT ego`) boosts performance even more. `COME-O` (with GT ego) achieves the highest average mIoU of 34.23%. Among models that predict the ego trajectory, `DTT-O` is a strong performer (30.85% mIoU).
*   **Analysis:** This highlights a key challenge: perception from raw camera feeds is still harder than predicting dynamics from a clean, structured representation like an occupancy grid. The performance drop over longer horizons (from 1s to 3s) demonstrates the difficulty of long-term prediction and error accumulation, even in state-level forecasting.

### 6.1.3. Control Tasks: DMC and RLBench (Table VI & VII)
These benchmarks evaluate how well a world model supports policy learning for control.
*   **DMC (Table VI):** The results show a trend of increasing **sample efficiency**. While early models like `PlaNet` required 5M steps, later models like `Dreaming` and `HRSSM` achieve strong performance with only 500k steps. This shows that architectural improvements (e.g., in `HRSSM` and `DreamerPro`) are leading to more efficient learning of control-relevant dynamics.
*   **RLBench (Table VII):** This benchmark tests complex robotic manipulation. The results indicate that recent models are leveraging more powerful backbones and multimodal inputs. `VidMan`, using an Inverse Dynamics Model (IDM) on top of a diffusion model, achieves a high average success rate (67%) across 18 tasks. Models using explicit 3D representations like `ManiGaussian` (3DGS) also show promise, though direct comparison is difficult due to varying task sets and evaluation protocols.
*   **Analysis:** For control, the trend is moving towards more sophisticated representations (beyond simple latent vectors) and architectures (like IDM and diffusion) that can better capture the fine-grained dynamics needed for manipulation.

### 6.1.4. Planning on nuScenes (Table VIII)
This task evaluates a model's ability to plan a safe and accurate trajectory for an autonomous vehicle. Key metrics are L2 error (accuracy) and collision rate (safety).
*   **Observation:** A clear trade-off between accuracy and safety emerges. $UniAD+DriveWorld$, which uses extensive auxiliary supervision (maps, object boxes, etc.), achieves the lowest L2 error (0.69m avg). However, `SSR`, which uses no auxiliary supervision, achieves the lowest collision rate (0.15% avg) while maintaining a competitive L2 error (0.75m).
*   **Analysis:** This suggests that optimizing purely for trajectory accuracy does not guarantee safety. The success of `SSR` indicates that clever self-supervised methods can learn very effective and safe driving policies even from raw camera inputs. The strong performance of camera-only methods in general shows that the field is maturing towards true end-to-end driving systems that do not rely on privileged information.

## 6.2. Data Presentation (Tables)
The following are the results from Table IV of the original paper:
**TABLE IV PERFORMANCE COMPARISON OF VIDEO GENERATION ON THE NUSCENES.**

| Method | Pub. | Resolution | FID↓ | FVD↓ |
| :--- | :--- | :--- | :--- | :--- |
| MagicDrive3D [84] | arXiv'24 | 224 × 400 | 20.7 | 164.7 |
| Delphi [86] | arXiv'24 | 512 × 512 | 15.1 | 113.5 |
| Drive-WM [88] | CVPR'24 | 192 × 384 | 15.8 | 122.7 |
| GenAD [90] | CVPR'24 | 256 × 448 | 15.4 | 184.0 |
| DriveDreamer [91] | ECCV'24 | 128 × 192 | 52.6 | 452.0 |
| Vista [96] | NeurIPS'24 | 576 × 1024 | 6.9 | 89.4 |
| DrivePhysica [214] | arXiv'24 | 256 × 448 | 4.0 | 38.1 |
| DrivingWorld [133] | arXiv'24 | 512 × 1024 | 7.4 | 90.9 |
| DriveDreamer-2 [97] | AAAI'25 | 256 × 448 | 11.2 | 55.7 |
| UniFuture [206] | arXiv'25 | 320 × 576 | 11.8 | 99.9 |
| MiLA [189] | arXiv'25 | 360 × 640 | 4.1 | 14.9 |
| GeoDrive [170] | arXiv'25 | 480 × 720 | 4.1 | 61.6 |
| LongDWM [188] | arXiv'25 | 480 × 720 | 12.3 | 102.9 |
| MaskGWM [104] | CVPR'25 | 288 × 512 | 8.9 | 65.4 |
| GEM [102] | CVPR'25 | 576 × 1024 | 10.5 | 158.5 |
| Epona [148] | ICCV'25 | 512 × 1024 | 7.5 | 82.8 |
| STAGE [198] | IROS'25 | 512 × 768 | 11.0 | 242.8 |
| DriVerse [109] | ACMMM'25 | 480 × 832 | 18.2 | 95.2 |

The following are the results from Table V of the original paper:
**TABLE V PERFORMANCE cOMPARISON OF 4D OCCUPANCY FORECASTING ON THE OCC3D-NUSCENES BENCHMARK1.**

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Input</th>
<th rowspan="2">Aux. Sup</th>
<th rowspan="2">Ego traj.</th>
<th colspan="5">mIoU (%) ↑</th>
<th colspan="5">IoU (%) ↑</th>
</tr>
<tr>
<th>Recon.</th>
<th>1s</th>
<th>2s</th>
<th>3s</th>
<th>Avg.</th>
<th>Recon.</th>
<th>1s</th>
<th>2s</th>
<th>3s</th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Copy &amp; Paste2</td>
<td>Occ</td>
<td>None</td>
<td>Pred.</td>
<td>66.38</td>
<td>14.91</td>
<td>10.54</td>
<td>8.52</td>
<td>11.33</td>
<td>62.29</td>
<td>24.47</td>
<td>19.77</td>
<td>17.31</td>
<td>20.52</td>
</tr>
<tr>
<td>OccWorld-O [93]</td>
<td>Occ</td>
<td>None</td>
<td>Pred.</td>
<td>66.38</td>
<td>25.78</td>
<td>15.14</td>
<td>10.51</td>
<td>17.14</td>
<td>62.29</td>
<td>34.63</td>
<td>25.07</td>
<td>20.18</td>
<td>26.63</td>
</tr>
<tr>
<td>OccLLaMA-O [18]</td>
<td>Occ</td>
<td>None</td>
<td>Pred.</td>
<td>75.20</td>
<td>25.05</td>
<td>19.49</td>
<td>15.26</td>
<td>19.93</td>
<td>63.76</td>
<td>34.56</td>
<td>28.53</td>
<td>24.41</td>
<td>29.17</td>
</tr>
<tr>
<td>RenderWorld-O [156]</td>
<td>Occ</td>
<td>None</td>
<td>Pred.</td>
<td>-</td>
<td>28.69</td>
<td>18.89</td>
<td>14.83</td>
<td>20.80</td>
<td>-</td>
<td>37.74</td>
<td>28.41</td>
<td>24.08</td>
<td>30.08</td>
</tr>
<tr>
<td>DTT-O [98]</td>
<td>Occ</td>
<td>None</td>
<td>Pred.</td>
<td>85.50</td>
<td>37.69</td>
<td>29.77</td>
<td>25.10</td>
<td>30.85</td>
<td>92.07</td>
<td>76.60</td>
<td>74.44</td>
<td>72.71</td>
<td>74.58</td>
</tr>
<tr>
<td>DFIT-OccWorld-O [174]</td>
<td>Occ</td>
<td>None</td>
<td>Pred.</td>
<td>-</td>
<td>31.68</td>
<td>21.29</td>
<td>15.18</td>
<td>22.71</td>
<td>-</td>
<td>40.28</td>
<td>31.24</td>
<td>25.29</td>
<td>32.27</td>
</tr>
<tr>
<td>COME-O [213]</td>
<td>Occ</td>
<td>None</td>
<td>Pred.</td>
<td>-</td>
<td>30.57</td>
<td>19.91</td>
<td>13.38</td>
<td>21.29</td>
<td></td>
<td>36.96</td>
<td>28.26</td>
<td>21.86</td>
<td>29.03</td>
</tr>
<tr>
<td>DOME-O [94]</td>
<td>Occ</td>
<td>None</td>
<td>GT</td>
<td>83.08</td>
<td>35.11</td>
<td>25.89</td>
<td>20.29</td>
<td>27.10</td>
<td>77.25</td>
<td>43.99</td>
<td>35.36</td>
<td>29.74</td>
<td>36.36</td>
</tr>
<tr>
<td>COME-O [213]</td>
<td>Occ</td>
<td>None</td>
<td>GT</td>
<td>-</td>
<td>42.75</td>
<td>32.97</td>
<td>26.98</td>
<td>34.23</td>
<td>-</td>
<td>50.57</td>
<td>43.47</td>
<td>38.36</td>
<td>44.13</td>
</tr>
<tr>
<td>OccWorld-T [93]</td>
<td>Camera</td>
<td>Semantic LiDAR</td>
<td>Pred.</td>
<td>7.21</td>
<td>4.68</td>
<td>3.36</td>
<td>2.63</td>
<td>3.56</td>
<td>10.66</td>
<td>9.32</td>
<td>8.23</td>
<td>7.47</td>
<td>8.34</td>
</tr>
<tr>
<td>OccWorld-S [93]</td>
<td>Camera</td>
<td>None</td>
<td>Pred.</td>
<td>0.27</td>
<td>0.28</td>
<td>0.26</td>
<td>0.24</td>
<td>0.26</td>
<td>4.32</td>
<td>5.05</td>
<td>5.01</td>
<td>4.95</td>
<td>5.00</td>
</tr>
<tr>
<td>RenderWorld-S [156]</td>
<td>Camera</td>
<td>None</td>
<td>Pred.</td>
<td>-</td>
<td>2.83</td>
<td>2.55</td>
<td>2.37</td>
<td>2.58</td>
<td>-</td>
<td>14.61</td>
<td>13.61</td>
<td>12.98</td>
<td>13.73</td>
</tr>
<tr>
<td>COME-S [213]</td>
<td>Camera</td>
<td>None</td>
<td>Pred.</td>
<td>-</td>
<td>25.57</td>
<td>18.35</td>
<td>13.41</td>
<td>19.11</td>
<td>-</td>
<td>45.36</td>
<td>37.06</td>
<td>30.46</td>
<td>37.63</td>
</tr>
<tr>
<td>OccWorld-D [93]</td>
<td>Camera</td>
<td>Occ</td>
<td>Pred.</td>
<td>18.63</td>
<td>11.55</td>
<td>8.10</td>
<td>6.22</td>
<td>8.62</td>
<td>22.88</td>
<td>18.90</td>
<td>16.26</td>
<td>14.43</td>
<td>16.53</td>
</tr>
<tr>
<td>OccWorld-F [93]</td>
<td>Camera</td>
<td>Occ</td>
<td>Pred.</td>
<td>20.09</td>
<td>8.03</td>
<td>6.91</td>
<td>3.54</td>
<td>6.16</td>
<td>35.61</td>
<td>23.62</td>
<td>18.13</td>
<td>15.22</td>
<td>18.99</td>
</tr>
<tr>
<td>OccLLaMA-F [18]</td>
<td>Camera</td>
<td>Occ</td>
<td>Pred.</td>
<td>37.38</td>
<td>10.34</td>
<td>8.66</td>
<td>6.98</td>
<td>8.66</td>
<td>38.92</td>
<td>25.81</td>
<td>23.19</td>
<td>19.97</td>
<td>22.99</td>
</tr>
<tr>
<td>DFIT-OccWorld-F [174]</td>
<td>Camera</td>
<td>Occ</td>
<td>Pred.</td>
<td>-</td>
<td>13.38</td>
<td>10.16</td>
<td>7.96</td>
<td>10.50</td>
<td>-</td>
<td>19.18</td>
<td>16.85</td>
<td>15.02</td>
<td>17.02</td>
</tr>
<tr>
<td>DTT-F [98]</td>
<td>Camera</td>
<td>Occ</td>
<td>Pred.</td>
<td>43.52</td>
<td>24.87</td>
<td>18.30</td>
<td>15.63</td>
<td>19.60</td>
<td>54.31</td>
<td>38.98</td>
<td>37.45</td>
<td>31.89</td>
<td>36.11</td>
</tr>
<tr>
<td>DOME-F [94]</td>
<td>Camera</td>
<td>Occ</td>
<td>GT</td>
<td>75.00</td>
<td>24.12</td>
<td>17.41</td>
<td>13.24</td>
<td>18.25</td>
<td>74.31</td>
<td>35.18</td>
<td>27.90</td>
<td>23.44</td>
<td>28.84</td>
</tr>
<tr>
<td>COME-F [213]</td>
<td>Camera</td>
<td>Occ</td>
<td>GT</td>
<td>-</td>
<td>26.56</td>
<td>21.73</td>
<td>18.49</td>
<td>22.26</td>
<td>-</td>
<td>48.08</td>
<td>43.84</td>
<td>40.28</td>
<td>44.07</td>
</tr>
</tbody>
</table>

The following are the results from Table VI of the original paper:
**TABLE VI PERFORMaNCE COMPARISON ON THE DMC BENCHMaRK.1.**

| Method | Step | Reacher Easy | Cheetah Run | Finger Spin | Walker Walk | Avg. / Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PlaNet [38] | 5M | 469 | 496 | 495 | 945 | 333/20 |
| Dreamer [10] | 5M | 935 | 895 | 499 | 962 | 823/20 |
| Dreaming [110] | 500k | 905 | 566 | 762 | 469 | 610/12 |
| TransDreamer [28] | 2M | - | 865 | - | 933 | 893/4 |
| DreamerPro [111] | 1M | 873 | 897 | 811 | - | 857/6 |
| MWM [41] | 1M | - | 670 | | - | 690/7 |
| HRSSM [25] | 500k | 910 | - | 960 | - | 938/3 |
| DisWM [112] | 1M | 960 | 820 | - | 920 | 879/5 |

The following are the results from Table VII of the original paper:
**TABLE VII PERFORMANCE COMPARISON FOR MANIPULATION TASKS ON RLBENCH.**

<table>
<thead>
<tr>
<th colspan="2">Criteria</th>
<th colspan="5">Methods</th>
</tr>
<tr>
<th colspan="2"></th>
<th>VidMan [55]</th>
<th>ManiGaussian [53]</th>
<th>ManiGaussian++ [80]</th>
<th>DreMa [60]</th>
<th>TesserAct [78]</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="6" valign="top">Settings</td>
<td>Episode</td>
<td>125</td>
<td>25</td>
<td>25</td>
<td>250</td>
<td>100</td>
</tr>
<tr>
<td>Pixel</td>
<td>224</td>
<td>128</td>
<td>256</td>
<td>128</td>
<td>512</td>
</tr>
<tr>
<td>Depth</td>
<td></td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Language</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td>✓</td>
</tr>
<tr>
<td>Proprioception</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Characteristic</td>
<td>IDM</td>
<td>GS</td>
<td>Bimanual</td>
<td>GS</td>
<td>DiT</td>
</tr>
<tr>
<td rowspan="6" valign="top">Success Rate (%)</td>
<td>Stack Blocks</td>
<td>48</td>
<td>12</td>
<td>-</td>
<td>12</td>
<td>-</td>
</tr>
<tr>
<td>Close Jar</td>
<td>88</td>
<td>28</td>
<td>-</td>
<td>51</td>
<td>44</td>
</tr>
<tr>
<td>Open Drawer</td>
<td>94</td>
<td>76</td>
<td>-</td>
<td>-</td>
<td>80</td>
</tr>
<tr>
<td>Sweep to Dustpan</td>
<td>93</td>
<td>64</td>
<td>92</td>
<td>-</td>
<td>56</td>
</tr>
<tr>
<td>Slide Block</td>
<td>98</td>
<td>24</td>
<td>-</td>
<td>62</td>
<td>-</td>
</tr>
<tr>
<td>Avg.1 / Total</td>
<td>67/18</td>
<td>45/10</td>
<td>35/10</td>
<td>25/9</td>
<td>63/10</td>
</tr>
</tbody>
</table>

The following are the results from Table VIII of the original paper:
**TABLE VIII PERFORMANCE COMPARISON FOR OPEN-LOOP PLANNING ON THE NUSCENES VALIDATION SPLIT1.**

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Input</th>
<th rowspan="2">Aux. Sup.2</th>
<th colspan="4">L2 (m) ↓</th>
<th colspan="4">Collision Rate (%) ↓</th>
</tr>
<tr>
<th>1s</th>
<th>2s</th>
<th>3s</th>
<th>Avg.</th>
<th>1s</th>
<th>2s</th>
<th>3s</th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>UniAD [254]</td>
<td>Camera</td>
<td>Map &amp; Box &amp; Motion &amp; Tracklets &amp; Occ</td>
<td>0.48</td>
<td>0.96</td>
<td>1.65</td>
<td>1.03</td>
<td>0.05</td>
<td>0.17</td>
<td>0.71</td>
<td>0.31</td>
</tr>
<tr>
<td>UniAD+DriveWorld [87]</td>
<td>Camera</td>
<td>Map &amp; Box &amp; Motion &amp; Tracklets &amp; Occ</td>
<td>0.34</td>
<td>0.67</td>
<td>1.07</td>
<td>0.69</td>
<td>0.04</td>
<td>0.12</td>
<td>0.41</td>
<td>0.19</td>
</tr>
<tr>
<td>GenAD [92]</td>
<td>Camera</td>
<td>Map &amp; Box &amp; Motion</td>
<td>0.36</td>
<td>0.83</td>
<td>1.55</td>
<td>0.91</td>
<td>0.06</td>
<td>0.23</td>
<td>1.00</td>
<td>0.43</td>
</tr>
<tr>
<td>FSDrive [101]</td>
<td>Camera</td>
<td>Map &amp; Box &amp; QA</td>
<td>0.40</td>
<td>0.89</td>
<td>1.60</td>
<td>0.96</td>
<td>0.07</td>
<td>0.12</td>
<td>1.02</td>
<td>0.40</td>
</tr>
<tr>
<td>OccWorld-T [93]</td>
<td>Camera</td>
<td>Semantic LiDAR</td>
<td>0.54</td>
<td>1.36</td>
<td>2.66</td>
<td>1.52</td>
<td>0.12</td>
<td>0.40</td>
<td>1.59</td>
<td>0.70</td>
</tr>
<tr>
<td>Doe-1 [134]</td>
<td>Camera</td>
<td>QA</td>
<td>0.50</td>
<td>1.18</td>
<td>2.11</td>
<td>1.26</td>
<td>0.04</td>
<td>0.37</td>
<td>1.19</td>
<td>0.53</td>
</tr>
<tr>
<td>SSR [160]</td>
<td>Camera</td>
<td>None</td>
<td>0.24</td>
<td>0.65</td>
<td>1.36</td>
<td>0.75</td>
<td>0.00</td>
<td>0.10</td>
<td>0.36</td>
<td>0.15</td>
</tr>
<tr>
<td>OccWorld-S [93]</td>
<td>Camera</td>
<td>None</td>
<td>0.67</td>
<td>1.69</td>
<td>3.13</td>
<td>1.83</td>
<td>0.19</td>
<td>1.28</td>
<td>4.59</td>
<td>2.02</td>
</tr>
<tr>
<td>Epona [148]</td>
<td>Camera</td>
<td>None</td>
<td>0.61</td>
<td>1.17</td>
<td>1.98</td>
<td>1.25</td>
<td>0.01</td>
<td>0.22</td>
<td>0.85</td>
<td>0.36</td>
</tr>
<tr>
<td>RenderWorld [156]</td>
<td>Camera</td>
<td>None</td>
<td>0.48</td>
<td>1.30</td>
<td>2.67</td>
<td>1.48</td>
<td>0.14</td>
<td>0.55</td>
<td>2.23</td>
<td>0.97</td>
</tr>
<tr>
<td>Drive-OccWorld [157]</td>
<td>Camera</td>
<td>None</td>
<td>0.32</td>
<td>0.75</td>
<td>1.49</td>
<td>0.85</td>
<td>0.05</td>
<td>0.17</td>
<td>0.64</td>
<td>0.29</td>
</tr>
<tr>
<td>OccWorld-D [93]</td>
<td>Camera</td>
<td>Occ</td>
<td>0.52</td>
<td>1.27</td>
<td>2.41</td>
<td>1.40</td>
<td>0.12</td>
<td>0.40</td>
<td>2.08</td>
<td>0.87</td>
</tr>
<tr>
<td>OccWorld-F [93]</td>
<td>Camera</td>
<td>Occ</td>
<td>0.45</td>
<td>1.33</td>
<td>2.25</td>
<td>1.34</td>
<td>0.08</td>
<td>0.42</td>
<td>1.71</td>
<td>0.73</td>
</tr>
<tr>
<td>OccLLaMA-F [18]</td>
<td>Camera</td>
<td>Occ</td>
<td>0.38</td>
<td>1.07</td>
<td>2.15</td>
<td>1.20</td>
<td>0.06</td>
<td>0.39</td>
<td>1.65</td>
<td>0.70</td>
</tr>
<tr>
<td>DTT-F [98]</td>
<td>Camera</td>
<td>Occ</td>
<td>0.35</td>
<td>1.01</td>
<td>1.89</td>
<td>1.08</td>
<td>0.08</td>
<td>0.33</td>
<td>0.91</td>
<td>0.44</td>
</tr>
<tr>
<td>DFIT-OccWorld-V [174]</td>
<td>Camera</td>
<td>Occ</td>
<td>0.42</td>
<td>1.14</td>
<td>2.19</td>
<td>1.25</td>
<td>0.09</td>
<td>0.19</td>
<td>1.37</td>
<td>0.55</td>
</tr>
<tr>
<td>NeMo [161]</td>
<td>Camera</td>
<td>Occ</td>
<td>0.39</td>
<td>0.74</td>
<td>1.39</td>
<td>0.84</td>
<td>0.00</td>
<td>0.09</td>
<td>0.82</td>
<td>0.30</td>
</tr>
<tr>
<td>OccWorld-O [93]</td>
<td>Occ</td>
<td>None</td>
<td>0.43</td>
<td>1.08</td>
<td>1.99</td>
<td>1.17</td>
<td>0.07</td>
<td>0.38</td>
<td>1.35</td>
<td>0.60</td>
</tr>
<tr>
<td>OccLLaMA-O [18]</td>
<td>Occ</td>
<td>None</td>
<td>0.37</td>
<td>1.02</td>
<td>2.03</td>
<td>1.14</td>
<td>0.04</td>
<td>0.24</td>
<td>1.20</td>
<td>0.49</td>
</tr>
<tr>
<td>RenderWorld-O [156]</td>
<td>Occ</td>
<td>None</td>
<td>0.35</td>
<td>0.91</td>
<td>1.84</td>
<td>1.03</td>
<td>0.05</td>
<td>0.40</td>
<td>1.39</td>
<td>0.61</td>
</tr>
<tr>
<td>DTT-O [98]</td>
<td>Occ</td>
<td>None</td>
<td>0.32</td>
<td>0.91</td>
<td>1.76</td>
<td>1.00</td>
<td>0.08</td>
<td>0.32</td>
<td>0.51</td>
<td>0.30</td>
</tr>
<tr>
<td>DFIT-OccWorld-O [174]</td>
<td>Occ</td>
<td>None</td>
<td>0.38</td>
<td>0.96</td>
<td>1.73</td>
<td>1.02</td>
<td>0.07</td>
<td>0.39</td>
<td>0.90</td>
<td>0.45</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
As a survey, this paper does not conduct its own ablation studies. Instead, its comprehensive comparison across different model types effectively serves as a high-level analysis of design choices. For example:
*   The performance gap between camera-only models and those using `Occ` (occupancy) or `LiDAR` inputs in Tables V and VIII reveals the impact of input modality.
*   The comparison between models with and without auxiliary supervision (`Aux. Sup.`) in Table VIII highlights the benefits and dependencies of using extra information like HD maps.
*   The evolution of scores on the DMC benchmark in Table VI from 5M steps to 500k steps is a clear indicator of how architectural changes have improved sample efficiency.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper provides a much-needed comprehensive survey of the world models field for embodied AI. Its primary contribution is a novel and intuitive three-axis taxonomy—`Functionality`, `Temporal Modeling`, and `Spatial Representation`—that unifies a diverse and fragmented body of research. By systematically organizing existing work, datasets, and metrics within this framework, the authors establish a clear standard for comparison and a valuable knowledge map for researchers. The paper concludes that while significant progress has been made, major challenges remain. These include the lack of unified, cross-domain datasets, the over-reliance on pixel-based metrics that ignore physical causality, and the core modeling trade-off between the efficiency of autoregressive models and the long-horizon stability of global prediction paradigms. Ultimately, the paper argues that advancing world models by developing hybrid methods that balance fidelity, efficiency, and interactivity is the key to building the next generation of intelligent embodied agents.

## 7.2. Limitations & Future Work
The paper distills several key challenges and corresponding future research directions from its analysis:

*   **Data & Evaluation:**
    *   **Challenge:** The field lacks large-scale, unified, multimodal datasets that span different domains (robotics, driving, etc.). Existing evaluation metrics like `FID` and `FVD` focus on pixel-level fidelity and fail to capture physical consistency, object permanence, or causal reasoning.
    *   **Future Work:** Prioritize the construction of unified, cross-domain benchmarks. Develop new evaluation frameworks that assess physical plausibility and causal understanding over simple visual realism.

*   **Computational Efficiency:**
    *   **Challenge:** High-performance models like Transformers and Diffusion Models are often too computationally expensive for real-time control on physical robots. This creates a trade-off between performance and efficiency.
    *   **Future Work:** Explore model optimization techniques like quantization and pruning. Investigate more efficient architectures like State Space Models (e.g., `Mamba`) that promise to combine long-range reasoning with linear-time complexity.

*   **Modeling Strategy:**
    *   **Challenge:** A core tension exists between `sequential simulation`, which suffers from error accumulation over long horizons, and `global prediction`, which is computationally heavy and less suited for interactive control. Similarly, spatial representations present a trade-off between the efficiency of abstract vectors and the high fidelity of explicit 3D rendering.
    *   **Future Work:** Develop hybrid methods that integrate the strengths of both temporal modeling paradigms. Improve long-horizon stability using techniques like explicit memory or hierarchical planning. Design unified architectures that seamlessly integrate temporal and spatial modeling to strike a better balance between fidelity, efficiency, and interactivity.

## 7.3. Personal Insights & Critique
This survey is an exceptional piece of work that serves as both an excellent introduction for newcomers and a valuable organizational tool for experts.

*   **Strengths:**
    *   The **three-axis taxonomy is the standout contribution**. It is clear, intuitive, and remarkably effective at structuring a chaotic field. It gives researchers a language to precisely describe and compare different world models.
    *   The paper is **extremely comprehensive and up-to-date**, covering a vast number of recent papers and providing a forward-looking perspective by including works from upcoming 2025 conferences.
    *   The consolidation of datasets and metrics is a highly valuable resource that will undoubtedly be cited frequently.

*   **Potential Issues and Areas for Improvement:**
    *   **The "Causality" Axis:** The paper correctly identifies physical consistency and causality as a weakness of current evaluation metrics. However, causality could almost be a fourth axis in the taxonomy itself. How a model represents and reasons about causal relationships is a fundamental design choice that is only implicitly covered by the current framework.
    *   **Short Half-Life:** As with any survey in a fast-moving field like AI, its detailed review of state-of-the-art models will inevitably become dated. However, the proposed taxonomy and the identified challenges are fundamental and will remain relevant for much longer. The accompanying GitHub repository is a smart way to mitigate this.
    *   **Generalist vs. Specialist Trade-off:** The paper touches on `Decision-Coupled` vs. `General-Purpose` models, but a deeper discussion on the trade-offs would be beneficial. Is the ultimate goal a single, massive "foundation world model" that can do everything, or will specialized, efficient models always be necessary for real-world robotics? The survey lays the groundwork for this debate but does not delve deeply into it.

        Overall, this paper is a landmark survey that provides immense clarity and structure to the field of world models. Its framework is likely to influence how researchers think about, design, and categorize world models for years to come. It is an essential read for anyone working in or entering the fields of embodied AI, robotics, and autonomous systems.