# 1. Bibliographic Information

## 1.1. Title
Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models

## 1.2. Authors
The paper was authored by Minting Pan, Xiangming Zhu, Yunbo Wang, and Xiaokang Yang. All authors are affiliated with the MoE Key Lab of Artificial Intelligence, AI Institute at Shanghai Jiao Tong University. The authors have strong backgrounds in deep learning, computer vision, and sequence modeling. Notably, Yunbo Wang is a key contributor to the PredRNN family of models, which are influential in the field of video prediction, indicating deep expertise in modeling spatiotemporal dynamics.

## 1.3. Journal/Conference
This paper was published at the **International Conference on Learning Representations (ICLR) in 2022**. ICLR is a premier, top-tier academic conference in the field of deep learning and artificial intelligence. Its acceptance criteria are highly rigorous, and publications in ICLR are considered significant contributions to the field.

## 1.4. Publication Year
2022

## 1.5. Abstract
The paper introduces **Iso-Dream**, a model-based reinforcement learning (MBRL) framework designed to improve upon the Dream-to-Control approach. The core problem it addresses is that in many real-world scenarios, such as autonomous driving, visual observations contain a mix of **controllable dynamics** (caused by the agent's actions) and **noncontrollable dynamics** (external factors like other vehicles). This mixture makes it hard for standard world models to learn effectively. Iso-Dream proposes a two-pronged solution. First, it introduces a world model that learns to **isolate** these two sources of dynamics into separate latent state branches, using an **inverse dynamics** objective to enforce the separation. Second, it proposes a new method for the agent to **leverage** this separation for decision-making. Specifically, the agent imagines future noncontrollable states (e.g., where other cars will be) and integrates this information with its current controllable state to make more forward-looking decisions. Experiments demonstrate that Iso-Dream successfully decouples the dynamics and significantly outperforms existing methods in various visual control and prediction tasks.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2205.13817
*   **PDF Link:** https://arxiv.org/pdf/2205.13817v3
*   **Publication Status:** The paper is a preprint available on arXiv and was officially published in the proceedings of ICLR 2022.

# 2. Executive Summary

## 2.1. Background & Motivation
The central problem this paper tackles lies at the intersection of **vision-based reinforcement learning** and the complexity of real-world environments. State-of-the-art Model-Based Reinforcement Learning (MBRL) methods, particularly "world models" like Dreamer, learn a simulator of the environment from pixels and then train an agent within this simulated or "imagined" world. This has led to significant improvements in sample efficiency.

However, a critical challenge remains: real-world visual scenes are not simple. They are often driven by multiple, entangled sources of change. For instance, in an autonomous driving scenario, the visual input changes due to two main factors:
1.  **Controllable Dynamics:** The agent (the ego-car) turns the wheel, and the scenery changes accordingly. This is a direct consequence of the agent's actions.
2.  **Noncontrollable Dynamics:** Other cars move on their own, pedestrians walk, and clouds drift in the sky. The agent has no control over these events, but they are crucial for safe and effective decision-making.

    Prior world models typically learn a single, monolithic representation of the environment's state, which conflates these different dynamic sources. This entanglement can corrupt the learned model, making it difficult for the agent to understand the true consequences of its actions and to plan effectively over long horizons.

The paper's innovative entry point is the explicit **disentanglement** of these dynamics based on the concept of **controllability**. The authors hypothesize that by forcing the world model to isolate what the agent can control from what it cannot, two key benefits can be achieved:
*   **Improved Robustness:** The model for controllable dynamics becomes robust to irrelevant, non-stationary noise (like a changing video background).
*   **Enhanced Long-Horizon Planning:** The agent can separately predict and reason about external events (like another car's trajectory) and use these predictions to make proactive, safer decisions.

## 2.2. Main Contributions / Findings
The paper makes several key contributions to the field of MBRL:

1.  **A Novel Disentangled World Model Architecture:** Iso-Dream introduces a world model with separate branches to model **controllable**, **noncontrollable**, and **static** components of a visual scene. This modular structure is the foundation for isolating different sources of dynamics.

2.  **Inverse Dynamics for Enforcing Disentanglement:** To ensure that the "controllable" branch truly captures only action-dependent changes, the model is trained with an **inverse dynamics objective**. This involves predicting the action that caused a given state transition, which forces the representation to be sensitive to control signals.

3.  **A "Visionary" Behavior Learning Algorithm:** The paper proposes a novel actor-critic algorithm that leverages the disentangled model. Instead of just considering the current state, the agent first "imagines" or rolls out the future evolution of the **noncontrollable dynamics** for several steps. It then uses an **attention mechanism** to integrate this preview of the future with its current controllable state, leading to more informed and forward-looking actions.

4.  **State-of-the-Art Performance:** The findings show that Iso-Dream significantly outperforms previous methods, including the strong DreamerV2 baseline. It achieves remarkable results on challenging benchmarks like the CARLA autonomous driving simulator and the DeepMind Control Suite with distracting dynamic backgrounds, demonstrating the practical benefits of its approach.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To fully grasp the paper, an understanding of the following concepts is essential:

*   **Reinforcement Learning (RL):** RL is a paradigm of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent takes an **action** in a given **state**, and the environment responds with a new state and a **reward** signal. The agent's goal is to learn a **policy** (a strategy for choosing actions) that maximizes the cumulative reward over time.

*   **Model-Based Reinforcement Learning (MBRL):** MBRL is a category of RL where the agent first learns a model of the environment's dynamics, often called a **world model**. This model predicts the next state and reward given the current state and an action ($s_{t+1}, r_t \sim M(s_t, a_t)$). Once this model is learned, the agent can use it to "plan" or "imagine" future outcomes without having to interact with the real, often costly, environment. This typically leads to much higher **sample efficiency** (learning from fewer real-world interactions) compared to model-free methods.

*   **World Models:** In the context of this paper, a world model is a deep learning model (often a recurrent neural network) that learns a compressed, low-dimensional **latent representation** of high-dimensional observations (like images) and models the transition dynamics within this latent space. This allows for efficient planning and imagination.

*   **Recurrent State-Space Model (RSSM):** This is the specific architecture used by Dreamer and Iso-Dream for their world models. An RSSM models the world state at each timestep $t$ using two components:
    *   A **deterministic hidden state** $h_t$, typically managed by a Recurrent Neural Network (RNN) like a GRU, which summarizes the history of past events.
    *   A **stochastic latent state** $s_t$, which is sampled from a distribution whose parameters are determined by the deterministic state $h_t$. This stochasticity helps model the inherent uncertainty in the environment.
        The transition is modeled as `h_t = f(h_{t-1}, s_{t-1}, a_{t-1})` and $s_t \sim p(s_t | h_t)$.

*   **Variational Autoencoder (VAE):** A VAE is a generative model that learns to encode high-dimensional data (like an image) into a low-dimensional latent space and then decode it back. It does this by optimizing the **Evidence Lower Bound (ELBO)**, a loss function that includes a reconstruction term (how well the decoded data matches the original) and a regularization term, typically the **Kullback-Leibler (KL) divergence**. The KL divergence encourages the learned latent distribution (the posterior) to be close to a simple prior distribution (like a standard Gaussian), which helps organize the latent space. The loss function in Iso-Dream (Eq. 6) is based on this principle.

*   **Inverse Dynamics:** This refers to the task of inferring the action $a_t$ that caused a transition from state $s_t$ to state $s_{t+1}$. In this paper, it is not used for control but as a powerful self-supervised learning signal. By forcing a part of the model to predict the action, it encourages that part to specifically learn and represent the aspects of the world that are affected by actions.

## 3.2. Previous Works
*   **Dreamer / Dream-to-Control [22]:** This is the direct foundation upon which Iso-Dream is built. Dreamer introduced the idea of learning a world model (an RSSM) and then training an actor-critic RL agent *entirely within the latent space* of this learned model. The agent learns by "dreaming" or imagining thousands of trajectories in parallel, making it highly efficient. However, Dreamer's world model uses a single latent state, which entangles all sources of dynamics. Iso-Dream explicitly addresses this limitation.

*   **PlaNet [23]:** A precursor to Dreamer, PlaNet also used an RSSM as a world model. However, instead of learning a policy via an actor-critic method, it performed online planning at each step using model-predictive control and optimization algorithms like the Cross-Entropy Method (CEM) over imagined trajectories.

*   **PhyDNet [19]:** This work is relevant as it also performs disentanglement for video prediction. However, its criterion for separation is different. PhyDNet attempts to separate dynamics that can be described by partial differential equations (PDEs), representing physical laws, from other unknown residual components. Iso-Dream's disentanglement is based on **controllability**, which is more directly relevant to an RL agent's decision-making process.

*   **InfoPower [4]:** This is a concurrent MBRL paper that also aims to improve representations. It uses an information-theoretic principle called "empowerment" to prioritize information from visual observations that is relevant to the agent's future actions. The authors of Iso-Dream differentiate their work by noting that Iso-Dream explicitly models the state transitions of noncontrollable dynamics and proposes a specific mechanism (future state attention) to leverage these predictions, which InfoPower does not.

## 3.3. Technological Evolution
The field of vision-based MBRL has seen a clear progression:
1.  **Early Models:** Initial attempts involved action-conditioned video prediction models that generated future frames directly in pixel space. Planning was then done over these predicted frames. These models were computationally expensive and often produced blurry, inaccurate long-term predictions.
2.  **Latent Dynamics Models (World Models, PlaNet):** A major breakthrough was learning a compressed latent space and modeling dynamics within it. This made long-term prediction more stable and computationally tractable. Planning was performed in this abstract latent space.
3.  **Learning Behaviors in Imagination (Dreamer):** The next step was to replace explicit planning with a learned policy (actor-critic). By training the agent entirely on imagined trajectories, Dreamer achieved high performance and sample efficiency.
4.  **Refining Latent Representations (Iso-Dream, InfoPower, etc.):** The current frontier, where Iso-Dream is situated, focuses on improving the quality and structure of the learned latent representations. Instead of a single monolithic state, methods like Iso-Dream argue for structured, disentangled representations that better reflect the causal nature of the world, leading to more robust and intelligent agents.

## 3.4. Differentiation Analysis
Compared to its main predecessor, **Dreamer**, Iso-Dream's core innovations are:
*   **Representation:** Dreamer uses a single latent state $(s_t)$ that mixes all dynamics. Iso-Dream uses two distinct latent states, a controllable state $(s_t)$ and a noncontrollable state $(z_t)$, plus a static background representation.
*   **Learning Objective:** Iso-Dream adds an **inverse dynamics loss** to explicitly force the controllable state $s_t$ to encode action-relevant information, a mechanism absent in Dreamer.
*   **Behavior Learning:** Dreamer's policy is conditioned only on the current latent state, $\pi(a_t | s_t)$. Iso-Dream's policy is conditioned on the current controllable state **and a forecast of future noncontrollable states**, $\pi(a_t | s_t, z_{t:t+\tau})$, enabling proactive decision-making.

# 4. Methodology

## 4.1. Principles
The core principle of Iso-Dream is that the complex dynamics observed in a visual scene can be more effectively modeled and utilized for control if they are first decomposed based on their source. The method assumes that the total dynamics $u_{1:T}$ can be separated into two primary components:
1.  **Controllable states $s_{1:T}$:** These are aspects of the environment that change as a direct result of the agent's actions $a_{1:T}$.
2.  **Noncontrollable states $z_{1:T}$:** These are aspects that evolve independently of the agent's actions.

    By isolating these components, the agent can build a cleaner model of action consequences ($s_t, a_t \rightarrow s_{t+1}$) and separately predict how the external world will evolve ($z_t \rightarrow z_{t+1}$). This separation then allows the agent to make more intelligent, forward-looking decisions by considering how its potential actions will interact with the predicted future state of the uncontrollable world.

The following figure from the paper provides a probabilistic graphical model illustrating this core idea.

![Figure 1: Probabilistic graph of Iso-Dream. It learns to decouple complex visual dynamics into controllable states $\\left( { { s _ { t } } } \\right)$ and noncontrollable states $\\left( { z _ { t } } \\right)$ by optimizing the inverse dynamics (Red dashed arrows). On top of the disentangled states, it performs model-based reinforcement learning by explicitly considering the predicted noncontrollable component of future dynamics (Blue arrows).](images/1.jpg)
*该图像是示意图，展示了Iso-Dream的概率图模型。图中通过优化逆动态（红色虚线箭头）将复杂的视觉动态解耦为可控状态 $s_t$ 和不可控状态 $z_t$。在解耦状态的基础上，模型基于强化学习，通过显式考虑未来动态的预测不可控成分（蓝色箭头）来增强决策能力。*

## 4.2. Core Methodology In-depth (Layer by Layer)
The Iso-Dream framework is composed of two interconnected parts: **Representation Learning**, which trains the disentangled world model, and **Behavior Learning**, which trains the agent's policy using this model.

### 4.2.1. Representation Learning of Controllable and Noncontrollable Dynamics

The world model in Iso-Dream is designed with a three-branch architecture to disentangle the visual data, as shown in Figure 2(a) of the paper.

![Figure 2: The overall architecture of the world model and the behavior learning algorithm in IsoDream. (a) World model with three branches to explicitly disentangle controllable, noncontrollable, and static components from visual data, where the action-conditioned branch learns controllable state transitions by modeling inverse dynamics. (b) The agent optimizes the behaviors in imaginations of the world model through a future state attention mechanism.](images/2.jpg)
*该图像是Iso-Dream世界模型及行为学习算法的示意图。 (a) 世界模型包含三个分支，明确区分可控、不可控和静态组件，其中动作条件分支通过建模逆动态来学习可控状态转移。 (b) 代理通过未来状态关注机制在世界模型的想象中优化行为。*

The model's objective is to learn the transitions of controllable and noncontrollable states. The paper formalizes this assumption as:
$$
u _ { 1 : T } \sim ( s , z ) _ { 1 : T } , \quad s _ { t + 1 } \sim p ( s _ { t + 1 } \mid s _ { t } , a _ { t } ) , \quad z _ { t + 1 } \sim p ( z _ { t + 1 } \mid z _ { t } ) ,
$$
where $s_t$ is the controllable state, $z_t$ is the noncontrollable state, and $a_t$ is the action.

**1. The Three-Branch Architecture:**

*   **Action-Conditioned Branch:** This branch models the controllable dynamics $p(s_{t+1} | s_t, a_t)$. It is based on the RSSM architecture from PlaNet and Dreamer. At each timestep, it updates a deterministic hidden state $h_t$ using a GRU and the previous states and action: `h_t = \mathtt{GRU}_s(h_{t-1}, s_{t-1}, a_{t-1})`. This $h_t$ is then used to define the distribution of the stochastic controllable state $s_t$.
*   **Action-Free Branch:** This branch models the noncontrollable dynamics $p(z_{t+1} | z_t)$. It has a similar RSSM structure but crucially takes no action input. Its deterministic state is updated as `h'_t = \mathtt{GRU}_z(h'_{t-1}, z_{t-1})`.
*   **Static Branch:** This branch is responsible for representing the time-invariant background. It encodes information from the first few frames of a sequence to generate a static representation.

    The prior distributions for the stochastic states are defined based on their respective deterministic histories:
$$
\begin{array} { r l } & { p ( \widetilde { s } _ { t } \mid s _ { < t } , a _ { < t } ) = p ( \widetilde { s } _ { t } \mid h _ { t } ) , \quad \mathrm { w h e r e } h _ { t } = \mathtt { G R U } _ { s } ( h _ { t - 1 } , s _ { t - 1 } , a _ { t - 1 } ) , } \\ & { \qquad p ( \widetilde { z } _ { t } \mid z _ { < t } ) = p ( \widetilde { z } _ { t } \mid h _ { t } ^ { \prime } ) , \quad \mathrm { w h e r e } h _ { t } ^ { \prime } = \mathtt { G R U } _ { z } ( h _ { t - 1 } ^ { \prime } , z _ { t - 1 } ) . } \end{array}
$$
Here, $\tilde{s}_t$ and $\tilde{z}_t$ denote the prior states predicted by the transition models. During training, these are compared against posterior states $s_t \sim q(s_t|h_t, o_t)$ and $z_t \sim q(z_t|h'_t, o_t)$, which are inferred using the current observation $o_t$.

**2. Inverse Dynamics for Disentanglement:**

To ensure the action-conditioned branch truly captures controllable dynamics, an `Inverse Cell` (a 2-layer MLP) is introduced. This cell takes two consecutive posterior controllable states, $s_{t-1}$ and $s_t$, and tries to predict the action $a_{t-1}$ that caused the transition.
$$
\tilde { \boldsymbol { a } } _ { t - 1 } = \mathtt { M L P } \big ( \boldsymbol { s } _ { t - 1 } , \boldsymbol { s } _ { t } \big ) ,
$$
The model is then trained to minimize the difference between the predicted action $\tilde{a}_{t-1}$ and the true action $a_{t-1}$. This loss signal forces the representation $s_t$ to contain information about the action's effects.

**3. Image Reconstruction:**

The three branches' outputs are combined to reconstruct the original image observation $o_t$. Each dynamic branch (controllable and noncontrollable) generates a visual component and a corresponding mask. The final image is a masked composition of the controllable component $\hat{o}_t^s$, the noncontrollable component $\hat{o}_t^z$, and the static background component $\hat{o}^b$.
$$
\hat { o } _ { t } = M _ { t } ^ { s } \odot \hat { o } _ { t } ^ { s } + M _ { t } ^ { z } \odot \hat { o } _ { t } ^ { z } + ( 1 - M _ { t } ^ { s } - M _ { t } ^ { z } ) \odot \hat { o } ^ { b } , \quad \mathrm { w h e r e ~ } \hat { o } ^ { b } = \mathtt { D e c } _ { \varphi _ { 3 } } \big ( \mathrm { E n c } _ { \theta , \phi _ { 3 } } \big ( o _ { 1 : K } \big ) \big ) \big ) .
$$
*   $M_t^s$ and $M_t^z$ are learned spatial masks.
*   $\odot$ denotes element-wise multiplication.
*   An important detail is that the controllable component $\hat{o}_t^s$ is reconstructed from the *prior* state $\tilde{s}_t$, while the noncontrollable component $\hat{o}_t^z$ is reconstructed from the *posterior* state $z_t$. This prevents the powerful action-conditioned branch from "cheating" by using information from the current observation to model all dynamics, thereby encouraging true separation.

**4. Overall Loss Function:**

The entire world model is trained end-to-end by maximizing the Evidence Lower Bound (ELBO), which translates to minimizing the following loss function over sequences of data from a replay buffer:
$$
\begin{array} { r l } & { \mathcal { L } = \mathrm { E } \{ \displaystyle \sum _ { t = 1 } ^ { T } \underbrace { - \ln p \left( o _ { t } \mid h _ { t } , s _ { t } , h _ { t } ^ { \prime } , z _ { t } \right) } _ { \mathrm { image~loss } } \underbrace { - \ln p \left( r _ { t } \mid h _ { t } , s _ { t } , h _ { t } ^ { \prime } , z _ { t } \right) } _ { \mathrm { reward~loss } } \underbrace { - \ln p \left( \gamma _ { t } \mid h _ { t } , s _ { t } , h _ { t } ^ { \prime } , z _ { t } \right) } _ { \mathrm { discount~loss } } } \\ & { \quad \quad + \underbrace { \alpha \ell _ { 2 } ( a _ { t } , \tilde { a } _ { t } ) } _ { \mathrm { action ~ loss } } + \underbrace { \beta _ { 1 } \mathrm { K L } [ q ( s _ { t } \mid h _ { t } , o _ { t } ) \mid p ( s _ { t } \mid h _ { t } ) ] + \beta _ { 2 } \mathrm { K L } [ q ( z _ { t } \mid h _ { t } ^ { \prime } , o _ { t } ) \mid p ( z _ { t } \mid h _ { t } ^ { \prime } ) ] } _ { \mathrm { KL~divergence } } \} . } \end{array}
$$
*   **Image Loss:** The reconstruction error for the image $o_t$.
*   **Reward Loss:** The error in predicting the reward $r_t$.
*   **Discount Loss:** The error in predicting the continuation probability $\gamma_t$ (1 for non-terminal states, 0 for terminal states).
*   **Action Loss:** The L2 loss for the inverse dynamics model.
*   **KL Divergence:** The regularization terms for both the controllable and noncontrollable latent states, balancing the information from the observation and the learned dynamics model.
*   $\alpha, \beta_1, \beta_2$ are hyperparameters that weight the different loss components.

### 4.2.2. Behavior Learning in Decoupled Imaginations

Once the world model is trained, it is used to train the agent's policy (actor) and value function (critic). This is where Iso-Dream leverages the disentangled states for long-horizon decision-making.

**1. Future State Attention:**

Instead of making a decision based solely on the current controllable state $\tilde{s}_t$, the agent first generates a forecast of the noncontrollable world. It rolls out the action-free branch for $\tau$ future steps to get a sequence of predicted noncontrollable states $\tilde{z}_{t:t+\tau}$. An attention mechanism is then used to integrate this future information with the current controllable state:
$$
\text{Future state attention: } e _ { t } = \mathrm { s o f t m a x } \big ( \tilde { s } _ { t } \tilde { z } _ { t : t + \tau } ^ { T } \big ) \tilde { z } _ { t : t + \tau } + \tilde { s } _ { t } .
$$
*   Here, the current controllable state $\tilde{s}_t$ acts as the "query".
*   The sequence of future noncontrollable states $\tilde{z}_{t:t+\tau}$ acts as both the "keys" and "values".
*   The softmax attention computes weights indicating which future noncontrollable states are most relevant to the current controllable state.
*   The weighted sum of future states creates a context vector, which is added to the original $\tilde{s}_t$ via a residual connection.
*   The resulting vector $e_t$ is a "visionary" state representation that encodes both the agent's current controllable state and a summary of relevant future external events.

**2. Actor-Critic Update:**

The action and value models are then updated to use this enhanced state representation $e_t$:
$$
\begin{align*} \mathrm { Action \ model: } \quad & a _ { t } \sim \pi ( a _ { t } \mid e _ { t } ) \\ \mathrm { Value \ model: } \quad & v _ { \xi } ( e _ { t } ) \approx \mathbb { E } _ { \pi ( \cdot \mid e _ { t } ) } \sum _ { k = t } ^ { t + L } \gamma ^ { k - t } r _ { k } \end{align*}
$$
*   The **Action model** (actor) learns a policy $\pi$ to select actions based on $e_t$.
*   The **Value model** (critic) learns to predict the expected future rewards from state $e_t$.
*   The training follows DreamerV2's approach of optimizing for the $\lambda$-return, which is an efficient way to propagate reward information over the imagination horizon $L$.

### 4.2.3. Policy Deployment and Overall Algorithm

The complete training loop is described in Algorithm 1 of the paper. It alternates between collecting new experience, training the world model, and training the agent in imagination.

When interacting with the real environment:
1.  The agent observes the current frame $o_t$ and encodes it into posterior states $s_t$ and $z_t$.
2.  It uses the action-free branch of its world model to predict the next $\tau-1$ noncontrollable states, $\tilde{z}_{t+1:t+\tau}$.
3.  It computes the "visionary" state $e_t$ using the future state attention mechanism with $s_t$ and the predicted noncontrollable states.
4.  It samples an action $a_t$ from its policy $\pi(a_t | e_t)$ and executes it in the environment.
5.  The resulting experience $(o_t, a_t, r_t, o_{t+1})$ is stored in the replay buffer.

    This process allows the agent to make decisions in the real world that are informed by its predictions of how uncontrollable elements (like other cars) will behave in the near future.

# 5. Experimental Setup

## 5.1. Datasets
Iso-Dream was evaluated on a diverse set of environments designed to test its core capabilities:

*   **DeepMind Control Suite (DMC) [46]:** The authors used a modified version from the DMC Generalization Benchmark [26] where tasks are performed with dynamic video backgrounds (`video_easy` setting). These backgrounds act as a source of **noncontrollable visual noise**. The tasks include `Walker Walk`, `Cheetah Run`, `Finger Spin`, and `Hopper Stand`. This setup is ideal for testing the model's ability to isolate task-relevant controllable dynamics from irrelevant distractions.

*   **CARLA [11]:** An open-source, high-fidelity simulator for autonomous driving research. The experiment was a highway driving task in "Town04" where the agent's goal is to drive as far as possible in 1000 steps without colliding with 30 other moving vehicles or barriers. In this setting, the **other vehicles represent the critical noncontrollable dynamics** that the agent must anticipate.

*   **BAIR Robot Pushing [13]:** A real-world dataset of a robot arm pushing objects. To test disentanglement, the authors augmented the dataset by adding **bouncing balls** to the scenes. The robot arm's movement is controllable (conditioned on actions), while the bouncing balls are predictable but noncontrollable.

*   **RoboNet [9]:** A large-scale dataset of videos from seven different robots. Similar to the BAIR setup, this dataset was also augmented with bouncing balls to create a mixed-dynamics environment for video prediction evaluation.

## 5.2. Evaluation Metrics
The evaluation metrics were chosen based on the task domain (RL control vs. video prediction).

### 5.2.1. For Reinforcement Learning (DMC, CARLA)
*   **Average Return (Score):** This is the standard metric in RL.
    1.  **Conceptual Definition:** It measures the performance of an agent by calculating the total accumulated reward over an episode, averaged over multiple episodes. A higher average return indicates a more effective policy.
    2.  **Mathematical Formula:** For a single episode (trajectory) $\tau = (s_0, a_0, r_0, s_1, ...)$, the return is $G = \sum_{t=0}^{T} \gamma^t r_t$, where $\gamma$ is a discount factor. The paper reports the undiscounted sum, which is often referred to as the "score". The final reported value is the mean of these scores across several evaluation episodes.

### 5.2.2. For Video Prediction (BAIR, RoboNet)
*   **Peak Signal-to-Noise Ratio (PSNR):**
    1.  **Conceptual Definition:** PSNR measures the quality of a reconstructed or predicted image by comparing it to the original, ground-truth image. It is based on the pixel-wise Mean Squared Error (MSE). A higher PSNR value indicates a better quality prediction with less error. It is measured in decibels (dB).
    2.  **Mathematical Formula:**
        \$
        \mathrm{PSNR} = 10 \cdot \log_{10}\left(\frac{\mathrm{MAX}_I^2}{\mathrm{MSE}}\right)
        \$
    3.  **Symbol Explanation:**
        *   $\mathrm{MAX}_I$ is the maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image).
        *   $\mathrm{MSE}$ is the Mean Squared Error between the ground-truth image $I$ and the predicted image $K$: `\mathrm{MSE} = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2`.

*   **Structural Similarity Index Measure (SSIM):**
    1.  **Conceptual Definition:** SSIM is an image quality metric designed to be more consistent with human visual perception than PSNR. Instead of just comparing pixel values, it measures the similarity between two images based on three components: luminance, contrast, and structure. Its value ranges from -1 to 1, where 1 indicates a perfect match.
    2.  **Mathematical Formula:**
        \$
        \mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
        \$
    3.  **Symbol Explanation:**
        *   $\mu_x$ and $\mu_y$ are the average pixel values of images $x$ and $y$.
        *   $\sigma_x^2$ and $\sigma_y^2$ are the variances of images $x$ and $y$.
        *   $\sigma_{xy}$ is the covariance of $x$ and $y$.
        *   $c_1 = (k_1 L)^2$ and $c_2 = (k_2 L)^2$ are two variables to stabilize the division with a weak denominator, where $L$ is the dynamic range of pixel values, and $k_1, k_2$ are small constants.

## 5.3. Baselines
Iso-Dream was compared against a strong set of existing methods:
*   **For Visual Control:**
    *   `DreamerV2` [24]: The state-of-the-art MBRL method that Iso-Dream directly builds upon.
    *   `CURL` [34]: A model-free RL method that uses contrastive learning for representation learning.
    *   `SVEA` [25]: A model-free method that uses data augmentation to improve stability and generalization.
    *   `SAC` [21]: A popular and powerful model-free off-policy actor-critic algorithm.
    *   `DBC` [59]: An MBRL method that learns representations invariant to task-irrelevant details without image reconstruction.
*   **For Video Prediction:**
    *   `SVG` [10]: A stochastic video generation model.
    *   `SA-ConvLSTM` [35]: A spatiotemporal prediction model using self-attention and ConvLSTMs.
    *   `PhyDNet` [19]: A model that disentangles dynamics based on physical principles (PDEs).

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. DeepMind Control Suite
The results on DMC with dynamic backgrounds test the model's robustness to distractions.

The following are the results from Table 1 of the original paper:

| TASK | SVEA | CURL | DBC* | DreamerV2 | Iso-Dream |
| :--- | :--- | :--- | :--- | :--- | :--- |
| WALKER WALK | 826 ± 65 | 443 ± 206 | 32 ± 7 | 655 ± 47 | **911 ± 50** |
| CHEETAH RUN | 178 ± 64 | 269 ± 24 | 15 ± 5 | 475 ± 159 | **659 ± 62** |
| FINGER SPIN | 562 ± 22 | 280 ± 50 | 1 ± 2 | 755 ± 92 | **800 ± 59** |
| HOPPER STAND | 6 ± 8 | 451 ± 250 | 5 ± 9 | 260 ± 366 | **746 ± 312** |

*   **Analysis:** Iso-Dream consistently and remarkably outperforms all baselines, including the strong DreamerV2. In tasks like `Hopper Stand`, the improvement is dramatic. This strongly suggests that its ability to isolate the controllable agent dynamics from the noncontrollable, noisy video background allows it to learn a much more stable and effective policy. The qualitative results in Figure 3 (left) visually confirm this: the model correctly separates the agent's body into the controllable component and the background (sea waves) into the noncontrollable component.

    The following figure (Figure 3 from the original paper) shows qualitative results on DMC and CARLA.

    ![Figure 3: Video prediction results on the DMC (left) and CARLA (right) benchmarks of Iso-Dream. For each sequence, we use the first 5 images as context frames. Iso-Dream successfully disentangles controllable and noncontrollable components.](images/3.jpg)
    *该图像是Iso-Dream在DMC（左）和CARLA（右）基准上的视频预测结果。图中显示了各种时间步（t=5, t=15, t=35, t=50和t=10, t=15, t=20）的真实和预测帧，展示了可控和不可控组件的分离效果。*

### 6.1.2. CARLA Autonomous Driving
This experiment tests the core hypothesis: that leveraging predictions of noncontrollable dynamics improves long-horizon decision-making.

The following figure (Figure 4 from the original paper) shows the performance curves.

![Figure 4: Performance with 3 seeds on the CARLA driving task. (a) Comparison of existing methods, in which Iso-Dream outperforms DreamerV2 by a large margin. (b) Ablation studies that can show the respective impact of optimizing the inverse dynamics (orange), rolling out noncontrollable states (green), and modeling the time-invariant information with a separate network branch (red).](images/4.jpg)
*该图像是图表，展示了在CARLA驾驶任务上不同方法的性能对比（a）和Iso-Dream的消融研究（b）。在（a）中，Iso-Dream明显优于DreamerV2。在（b）中，显示了优化逆动态、滚动非可控状态和时间不变建模对表现的影响。*

*   **Analysis:** Figure 4(a) shows that Iso-Dream learns significantly faster and achieves a much higher final score than DreamerV2 and other methods. This demonstrates a substantial advantage in the complex autonomous driving scenario. The ability to anticipate the movements of other vehicles (noncontrollable dynamics) allows the Iso-Dream agent to make safer, more proactive maneuvers, leading to longer survival times and higher rewards.

### 6.1.3. BAIR & RoboNet Video Prediction
These experiments evaluate the quality of the world model's disentangled predictions.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">MODEL</th>
<th colspan="2">BAIR</th>
<th colspan="2">RoboNET</th>
</tr>
<tr>
<th>PSNR ↑</th>
<th>SSIM↑</th>
<th>PSNR ↑</th>
<th>SSIM↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>SVG [10]</td>
<td>18.12</td>
<td>0.712</td>
<td>19.86</td>
<td>0.708</td>
</tr>
<tr>
<td>SA-CONvLSTM [35]</td>
<td>18.28</td>
<td>0.677</td>
<td>19.30</td>
<td>0.638</td>
</tr>
<tr>
<td>PhyDNet [19]</td>
<td>18.91</td>
<td>0.743</td>
<td>20.89</td>
<td>0.727</td>
</tr>
<tr>
<td>Iso-Dream</td>
<td><strong>19.51</strong></td>
<td><strong>0.768</strong></td>
<td><strong>21.71</strong></td>
<td><strong>0.769</strong></td>
</tr>
</tbody>
</table>

*   **Analysis:** Iso-Dream achieves the best performance on both datasets according to both PSNR and SSIM metrics. This indicates that its world model produces higher-fidelity and more structurally accurate long-term video predictions. The qualitative results in Figure 5 visually support this, showing that the model correctly assigns the robot arm to the action-conditioned branch and the bouncing balls to the action-free branch.

    The following figure (Figure 5 from the original paper) shows qualitative results on the BAIR dataset.

    ![Figure 5: Showcases of video prediction results on the BAIR robot pushing dataset. We display every 3 frames in the prediction horizon. The generated masks show that each branch of Iso-Dream captures coarse localisation of controllable representations and noncontrollable representations.](images/5.jpg)
    *该图像是视频预测结果的展示，包含BAIR机器人推送数据集中的真实帧和Iso-Dream模型生成的帧。不同时间步的预测结果显示了可控和不可控表示的粗略定位，具体包括动作自由分支和动作条件分支的对比。*

## 6.2. Ablation Studies / Parameter Analysis
The ablation studies are crucial for validating that each proposed component of Iso-Dream contributes to its performance.

*   **CARLA Ablation (Figure 4b):**
    *   **"w/o Inverse Cell" (orange curve):** Removing the inverse dynamics objective causes a significant performance drop. This confirms that this loss is critical for forcing the desired disentanglement and ensuring the controllable branch is truly action-sensitive.
    *   **"w/o Rolling-out" (green curve):** This version replaces the future state attention with a simpler concatenation of the current controllable and noncontrollable states. The performance drop is substantial, proving that the key benefit comes from *predicting the future* of noncontrollable dynamics and using the attention mechanism to integrate this forecast.
    *   **"w/o Static branch" (red curve):** Removing the dedicated branch for the static background also degrades performance. This shows that explicitly modeling all three components (controllable, noncontrollable, static) is beneficial.

*   **BAIR Video Prediction Ablation (Table 3):**
    The following are the results from Table 3 of the original paper:

    | MODEL | Predict 18 Frames PSNR ↑ | SSIM ↑ | Predict 28 Frames PSNR ↑ | SSIM ↑ |
    | :--- | :--- | :--- | :--- | :--- |
    | Iso-Dream w/o action-free Branch | 20.47 | 0.795 | 18.51 | 0.690 |
    | Iso-Dream w/o Inverse Cell | 21.42 | 0.829 | 19.34 | 0.759 |
    | Iso-Dream | **21.43** | **0.832** | **19.51** | **0.768** |

*   **Analysis:** This table reinforces the findings from the CARLA ablation. Removing the action-free branch (i.e., reverting to a single dynamic branch) significantly hurts prediction quality. Removing the Inverse Cell also leads to a drop in performance, especially on the longer prediction horizon (28 frames), again highlighting its importance for learning a clean, disentangled representation.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully presents **Iso-Dream**, a novel and effective MBRL framework that addresses the challenge of mixed dynamics in complex visual environments. Its core contributions are twofold:
1.  It introduces a principled method for **learning disentangled representations** of controllable and noncontrollable dynamics, using a modular architecture regularized by an inverse dynamics objective.
2.  It proposes an innovative **behavior learning algorithm** that leverages this disentanglement by using an attention mechanism to reason about the predicted future of noncontrollable events, leading to more intelligent and forward-looking decisions.

    The comprehensive experiments on diverse and challenging benchmarks, especially the significant gains in the CARLA autonomous driving task, strongly validate the central hypothesis of the paper: isolating and leveraging noncontrollable dynamics is a powerful approach for building more capable and robust vision-based agents.

## 7.2. Limitations & Future Work
The authors candidly acknowledge two main limitations:

*   **Computational Efficiency:** The behavior learning phase in Iso-Dream is more computationally intensive than in DreamerV2 because it requires rolling out the noncontrollable dynamics branch at each step of the imagination. While the method is more sample-efficient (requiring fewer real-world interactions), it demands more training time per episode.
*   **Environment-Specific Architecture:** The authors note that the model's architecture and hyperparameters required some manual tuning based on prior knowledge of each environment (e.g., whether to use the noncontrollable states for reward prediction). This suggests that the framework is not yet a fully "plug-and-play" solution and that developing more adaptive architectures is a key direction for future work.

## 7.3. Personal Insights & Critique
*   **Strengths:** The paper's core idea is both intuitive and powerful. The framing of disentanglement around "controllability" is highly relevant for RL, as it directly relates to the agent's sense of agency and planning. The "future state attention" mechanism is an elegant and effective way to operationalize this disentangled knowledge. The strong empirical results, particularly in CARLA, provide compelling evidence for the method's real-world potential.

*   **Potential Issues & Unverified Assumptions:**
    *   The model assumes a clean separation between controllable and noncontrollable dynamics. However, in many real-world scenarios, these are coupled. For example, in autonomous driving, other drivers (noncontrollable) will react to the agent's actions (controllable). The current model treats other vehicles as evolving independently, which is a strong simplification. It is unclear how Iso-Dream would handle these interactive scenarios.
    *   The effectiveness of the inverse dynamics objective might depend on the action space. For high-dimensional or very subtle actions, predicting the action from state changes could be extremely difficult, potentially weakening the disentanglement signal.

*   **Inspirations and Future Directions:**
    *   **Multi-Agent RL:** The Iso-Dream framework could be a powerful foundation for multi-agent systems. Each agent could model other agents as noncontrollable (but predictable) dynamic components, using future state attention to anticipate their actions and plan accordingly.
    *   **Hierarchical RL:** The concept of controllability could be extended to a hierarchy. An agent could learn what is controllable at a low level (e.g., motor torques) versus a high level (e.g., navigating to a room), potentially leading to more abstract and efficient planning.
    *   **Online Adaptation:** Future work could focus on making the model more adaptive, perhaps by dynamically inferring which parts of the environment are controllable at any given time, removing the need for environment-specific design choices. This would be a significant step towards more general and autonomous intelligence.