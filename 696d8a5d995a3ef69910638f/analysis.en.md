# 1. Bibliographic Information

## 1.1. Title
Discrete Codebook World Models for Continuous Control

## 1.2. Authors
Mohammadreza Nakhaei (Aalto University), Aidan Scannell (University of Edinburgh), Kalle Kujanpää (Aalto University), Yi Zhao (Aalto University), Kevin Sebastian Luck (Vrije Universiteit Amsterdam), Arno Solin (Aalto University), Joni Pajarinen (Aalto University). The authors are primarily affiliated with Aalto University, a prominent research institution in Finland, with collaborations from the University of Edinburgh and Vrije Universiteit Amsterdam. Their research backgrounds focus on machine learning, reinforcement learning, and probabilistic modeling.

## 1.3. Journal/Conference
The paper is available as a preprint on arXiv. The publication date (`2025-03-01T22:58:44.000Z`) suggests it is intended for a future conference. Top-tier conferences for this type of research include the International Conference on Learning Representations (ICLR), the International Conference on Machine Learning (ICML), and the Conference on Neural Information Processing Systems (NeurIPS). These venues are highly competitive and influential in the field of machine learning.

## 1.4. Publication Year
2025 (as per the arXiv submission date).

## 1.5. Abstract
The paper investigates the use of world models with discrete latent spaces for state-based continuous control tasks in reinforcement learning (RL). While previous models with discrete spaces (like `DreamerV3`) have excelled in visual or discrete-action tasks, and models with continuous spaces (like `TD-MPC2`) have dominated state-based continuous control, this work bridges the gap. The authors demonstrate that discrete latent states, specifically those represented by codes from a codebook, are more effective than continuous states or other discrete encodings (one-hot, label). They introduce the **Discrete Codebook World Model (DCWM)**, which features a discrete and stochastic latent space. When combined with Model Predictive Control (MPC) for planning, the resulting algorithm, **Discrete Codebook Model Predictive Control (DC-MPC)**, achieves competitive performance against state-of-the-art methods like `TD-MPC2` and `DreamerV3` on challenging continuous control benchmarks.

## 1.6. Original Source Link
- **Original Source Link:** https://arxiv.org/abs/2503.00653
- **PDF Link:** https://arxiv.org/pdf/2503.00653v1
- **Publication Status:** This is a preprint on arXiv, meaning it has not yet undergone formal peer review for publication in a conference or journal.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem this paper addresses is how to design the most effective latent space for world models in the context of **state-based continuous control** reinforcement learning. A world model is an internal simulator learned by an RL agent to predict how its environment will change in response to its actions. This allows the agent to "imagine" future outcomes and plan more effectively.

A key design choice in world models is the nature of the latent space—the compressed representation of the environment's state. There has been a divergence in the field:
*   **Continuous Latent Spaces:** Methods like `TD-MPC2` use continuous vectors to represent states. They have shown state-of-the-art performance on benchmarks where the agent receives low-dimensional state information (e.g., joint angles of a robot) and must produce continuous actions (e.g., motor torques).
*   **Discrete Latent Spaces:** Methods like `DreamerV3` use discrete representations (specifically, one-hot vectors). They have been very successful in visually complex domains (e.g., learning from pixels) or tasks with discrete actions. However, their performance in state-based continuous control has lagged behind continuous-space models.

    This creates a research gap: **Can the benefits of discrete latent spaces be effectively translated to state-based continuous control, and if so, what is the best way to represent these discrete states?** The authors are motivated by the hypothesis that discrete spaces could offer advantages like preventing compounding prediction errors and enabling more efficient learning, but the traditional `one-hot` encoding used by `Dreamer` might not be optimal for representing inherently continuous state information. The paper's entry point is to explore a different kind of discrete representation: a **codebook**, which can preserve ordinal relationships in the data while still being discrete.

## 2.2. Main Contributions / Findings
The paper makes three primary contributions, which directly answer the questions raised by its motivation:

1.  **(C1) Discrete Latent Spaces are Beneficial for Continuous Control:** The authors experimentally demonstrate that learning a discrete latent space using a classification objective (predicting the next discrete state) outperforms learning a continuous latent space with a regression objective (predicting the next continuous state vector). This challenges the prevailing success of continuous-space models in this specific domain.

2.  **(C2) Codebook Encodings are Superior to Alternatives:** The paper shows that representing discrete states as vectors (`codes`) from a learned or fixed `codebook` is more effective than other discrete representations.
    *   It outperforms `one-hot` encoding (used in $DreamerV2/V3$), which treats all states as equidistant and fails to capture the underlying continuous nature of the state space.
    *   It outperforms `label` encoding (e.g., using integers 1, 2, 3...), which imposes a simple, one-dimensional ordering that is often insufficient for complex, multi-dimensional state spaces.

3.  **(C3) A New State-of-the-Art Algorithm (DC-MPC):** Based on these findings, the paper introduces a new model-based RL algorithm:
    *   **DCWM (Discrete Codebook World Model):** A world model with a discrete, stochastic latent space where states are represented by codes from a codebook generated via Finite Scalar Quantization (FSQ).
    *   **DC-MPC (Discrete Codebook Model Predictive Control):** The full algorithm, which combines `DCWM` with a decision-time planning method (MPPI) to select actions.
    *   **Performance:** `DC-MPC` achieves performance that is competitive with, and in some complex high-dimensional tasks even superior to, state-of-the-art methods like `TD-MPC2` and `DreamerV3` on standard continuous control benchmarks like DeepMind Control Suite, Meta-World, and MyoSuite.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Reinforcement Learning (RL)
Reinforcement Learning is a paradigm of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent takes an **action**, the environment transitions to a new **state**, and the agent receives a **reward** (or penalty). The agent's goal is to learn a **policy** (a strategy for choosing actions) that maximizes the cumulative reward over time. This process is often modeled as a **Markov Decision Process (MDP)**.

### 3.1.2. Model-Based vs. Model-Free RL
*   **Model-Free RL:** The agent learns a policy or a value function (which estimates the expected future reward) directly from experience, without trying to understand the environment's rules. It's like learning to ride a bike purely by trial and error. `SAC` (Soft Actor-Critic) is a prominent model-free algorithm.
*   **Model-Based RL:** The agent first learns a **world model** of the environment. This model acts as an internal simulator, predicting the next state and reward given the current state and an action ($s_{t+1}, r_t \approx \text{model}(s_t, a_t)$). The agent can then use this model to "plan" or "imagine" sequences of actions to find the best one, often leading to better **sample efficiency** (learning from fewer real-world interactions). This paper's method, `DC-MPC`, is model-based.

### 3.1.3. Latent Space
In machine learning, a latent space is a lower-dimensional, compressed representation of high-dimensional data. For world models, instead of predicting the full, raw observation (like an image or a long vector of sensor readings), the model first **encodes** the observation into a smaller latent state vector ($z_t$). The world model then predicts transitions in this compressed space ($z_{t+1} \approx \text{dynamics\_model}(z_t, a_t)$). This is computationally cheaper and can help the model focus on the most important features of the environment.

### 3.1.4. Discrete vs. Continuous Latent Spaces
*   **Continuous:** The latent state $z_t$ is a vector of real numbers (e.g., $z_t \in \mathbb{R}^{32}$). This is natural for representing continuous physical properties but can suffer from "compounding errors," where small prediction inaccuracies accumulate over long imagined sequences.
*   **Discrete:** The latent state $z_t$ can only take one of a finite number of values from a set. This can make learning more stable and robust to small errors, as the model is forced to snap its prediction to the nearest valid state.

### 3.1.5. Discrete Encodings
The paper discusses three ways to represent discrete information for a neural network:
*   **Label Encoding:** Assigns a unique integer to each category (e.g., A=1, B=2, C=3). This implicitly imposes an ordinal relationship ($A<B<C$) which may not be meaningful.
*   **One-Hot Encoding:** Represents each category with a binary vector that is all zeros except for a single one (e.g., A=[1,0,0], B=[0,1,0], C=[0,0,1]). This treats all categories as distinct and equidistant, which is good for nominal data (like types of fruit) but bad for continuous data where some states are "closer" to each other than others.
*   **Codebook Encoding:** Assigns a unique dense vector (a "code") from a predefined set (the "codebook") to each category (e.g., A=[-0.5, -0.5], B=[0,0], C=[0.5, 0.5]). This is a powerful hybrid: it's discrete (only a finite number of codes exist), but the codes themselves are vectors whose distances can represent relationships between the original data points (e.g., preserving ordinality). This paper argues this is the best approach for discretizing continuous control states.

### 3.1.6. Model Predictive Control (MPC)
MPC is a planning algorithm used for decision-making. At each time step, the agent:
1.  Uses its world model to simulate many possible action sequences over a short future horizon.
2.  Evaluates each sequence to see which one leads to the best cumulative reward.
3.  Executes only the *first* action of the best sequence.
4.  Observes the new state from the real environment and repeats the whole process.
    This makes the agent constantly re-plan and adapt to new information. The specific planning method used in this paper is `MPPI` (Model Predictive Path Integral).

## 3.2. Previous Works

### 3.2.1. Dreamer Family (DreamerV1, V2, V3)
*   **Core Idea:** The `Dreamer` algorithms (Hafner et al., 2019a, 2022, 2023) are state-of-the-art model-based RL agents. They learn a world model and then train an actor-critic agent entirely within the "imagination" of this model.
*   **Latent Space Evolution:**
    *   `DreamerV1` used a **continuous, stochastic** latent space modeled with a Gaussian distribution.
    *   $DreamerV2/V3$ made a crucial switch to a **discrete latent space** using **one-hot encoding**. This significantly improved performance, especially in visually complex domains.
*   **Learning Objective:** A key component of `Dreamer` is its reliance on an **observation reconstruction** loss. The model must be able to decode the latent state back into the original observation (e.g., reconstruct the image). This paper argues that reconstruction is detrimental for state-based control.

### 3.2.2. TD-MPC Family (TD-MPC, TD-MPC2)
*   **Core Idea:** `TD-MPC` and its successor `TD-MPC2` (Hansen et al., 2022, 2023) are powerful model-based RL agents designed for continuous control. They combine a learned world model with MPC for planning.
*   **Latent Space:** They use a **continuous, deterministic** latent space.
*   **Learning Objective:** Crucially, they are **decoder-free**. They do not use an observation reconstruction loss. Instead, they learn their latent representations using a **latent-state consistency** loss: the latent state predicted by the dynamics model should match the latent state encoded from the *actual* next observation. They also use value prediction to help structure the latent space.
*   **Performance:** `TD-MPC2` is considered the state of the art in many state-based continuous control benchmarks, significantly outperforming `DreamerV3`.

## 3.3. Technological Evolution
The evolution of world models in RL has seen a progression in how latent spaces are designed:
1.  **Early Models:** Focused on continuous, often deterministic, latent spaces.
2.  **DreamerV1:** Introduced a continuous but *stochastic* (probabilistic) latent space to better model uncertainty.
3.  **DreamerV2/V3:** Showed the power of *discrete* latent spaces (via one-hot encoding) for handling complex visual data and improving long-term stability.
4.  **TD-MPC2:** Demonstrated that for *state-based* control, a simple continuous, deterministic space learned with a consistency loss (and without a decoder) can be extremely effective.

    This paper's work fits into this timeline by asking: can we combine the best of both worlds? Can we use a discrete latent space (like $DreamerV2/V3$) but learn it with a consistency objective (like `TD-MPC2`) and use a more suitable encoding (`codebook`) for continuous control tasks?

## 3.4. Differentiation Analysis
The core innovations of `DC-MPC` compared to its main predecessors are:
*   **vs. DreamerV3:**
    *   **Encoding:** `DC-MPC` uses a `codebook` encoding instead of `one-hot`. This allows it to capture ordinal relationships in the continuous state data.
    *   **Learning Objective:** `DC-MPC` uses a latent-state consistency loss and is decoder-free, whereas `DreamerV3` relies heavily on observation reconstruction.
    *   **Planning:** `DC-MPC` uses decision-time planning (MPC), while `DreamerV3` primarily trains its policy in imagination and then executes it without online planning.

*   **vs. TD-MPC2:**
    *   **Latent Space:** `DC-MPC` uses a **discrete and stochastic** latent space, whereas `TD-MPC2` uses a **continuous and deterministic** one.
    *   **Learning Objective:** Because its latent space is discrete, `DC-MPC` trains its dynamics model with a **classification** objective (cross-entropy loss), which can model a multi-modal distribution over possible next states. `TD-MPC2` uses a **regression** objective (mean squared error), which assumes a single, unimodal prediction.
    *   **Value Prediction:** `DC-MPC` decouples representation learning from value learning, whereas `TD-MPC2` uses value prediction as part of the objective for training its encoder.

# 4. Methodology

## 4.1. Principles
The core idea of the proposed method, **Discrete Codebook Model Predictive Control (DC-MPC)**, is to build a world model that operates in a **discrete latent space** where each state is represented by a vector (`code`) from a fixed `codebook`. This design is motivated by the hypothesis that discretization provides robustness, while the vector-based codes retain crucial structural information about the original continuous state space. The world model is trained in a self-supervised manner to predict the probability distribution over the next discrete code, using a classification loss. This trained model is then used by a planner at decision time to select the best actions.

## 4.2. Core Methodology In-depth (Layer by Layer)
The `DC-MPC` algorithm has two main parts: (i) learning the world model, named **Discrete Codebook World Model (DCWM)**, and (ii) using this model for decision-time planning with Model Predictive Path Integral (`MPPI`).

### 4.2.1. The Discrete Codebook World Model (DCWM)
The `DCWM` consists of several interconnected neural network components. The paper lists six, but the core world model itself involves the first three, while the others are for policy and value learning.

1.  **Encoder ($e_{\theta}$):** This network takes a raw observation from the environment $o \in \mathcal{O}$ and maps it to a continuous latent vector $x$.
    \$
    x = e_{\theta}(o) \in \mathbb{R}^{b \times d}
    \$
    Here, $d$ is the latent dimension and $b$ is the number of channels, which are hyperparameters. For instance, an observation might be mapped to a tensor of shape `[2, 512]`.

2.  **Quantizer ($f$):** This is the key step that discretizes the representation. The continuous vector $x$ is mapped to a discrete code $c$ from a codebook $\mathcal{C}$. This is achieved using **Finite Scalar Quantization (FSQ)**.
    \$
    c = f(x) \in \mathcal{C}
    \$
    FSQ works by quantizing each of the $b \times d$ values in $x$ to a nearby integer level. The paper defines a set of quantization levels $\mathcal{L} = \{L_1, L_2, ..., L_b\}$, where $L_i$ is the number of discrete symbols for the $i$-th channel. The quantization function for the $i$-th channel is given by:
    \$
    f : x, \mathcal{L}, i \to \mathrm{round} \left( \left\lfloor \frac{L_i}{2} \right\rfloor \cdot \tanh(x_{i,:}) \right)
    \$
    *   $x_{i,:}$ refers to the vector of $d$ dimensions for the $i$-th channel.
    *   $\tanh(x_{i,:})$ squashes the continuous values into the range $[-1, 1]$.
    *   $\lfloor L_i/2 \rfloor$ scales this range. For example, if $L_i=5$, the scaler is 2. The range becomes $[-2, 2]$.
    *   `round(...)` snaps the result to the nearest integer. For $L_i=5$, the possible integer symbols would be $\{-2, -1, 0, 1, 2\}$.
        The total number of unique codes in the codebook is $|\mathcal{C}| = \prod_{i=1}^{b} L_i$. For example, with levels $\mathcal{L}=\{5,3\}$, there are $5 \times 3 = 15$ unique 2-dimensional codes per latent dimension $d$. This process creates a structured grid of code vectors, as shown in Figure 2.

    The following figure (Figure 2 from the original paper) visualizes how FSQ creates a multi-dimensional codebook.

    ![Figure 2: Illustration of Codebook $( \\mathcal { C } )$ FSQ's codebook is a $b$ -dimensional hypercube (left). This figure illustrates a $b { = } 3$ -dimensional codebook, where each axis of the 3-dimensional hypercube (left) corresponds to one dimension of the codebook (right). The $i ^ { \\mathrm { { t h } } }$ dimension of the hypercube is discretized into `L _ { i }` values, e.g., the $x$ and $y \\cdot$ -axis are discretized into `L _ { 0 } = L _ { 1 } = 5` and the $z$ -axis into $L _ { 3 } = 4$ . Code symbols (here integers) are normalized to the range $\[ - 1 , 1 \]$ .](images/2.jpg)
    *该图像是一个示意图，展示了一个$b=3$维的代码本（Codebook）`ext{C}`。每个坐标轴分别表示不同的维度，$x$和$y$轴被离散化为$L_{0}=L_{1}=5$个值，$z$轴为$L_{3}=4$个值。代码符号（整数）被规范化到区间$[-1, 1]$。*

    Since the `round` function is not differentiable, the authors use the **straight-through gradient estimator (STE)** to allow gradients to flow back to the encoder during training.

3.  **Dynamics Model ($p_{\phi}$):** This network predicts the transition dynamics within the discrete latent space. Given a current discrete code $c$ and an action $a$, it outputs a probability distribution over all possible next codes $c'$ in the codebook.
    \$
    c' \sim \mathrm{Categorical}(p_1, \dots, p_{|\mathcal{C}|}) \quad \text{with } p_i = p_{\phi}(c' = c^{(i)} | c, a)
    \$
    *   $c^{(i)}$ is the $i$-th code in the codebook $\mathcal{C}$.
    *   The model is implemented as a classifier. It takes `(c, a)` as input and outputs logits $l \in \mathbb{R}^{|\mathcal{C}|}$. The probabilities $p_i$ are then obtained by applying a `softmax` function to these logits.

4.  **Reward Predictor ($R_{\xi}$):** Predicts the immediate reward given a latent code and an action.
    \$
    r = R_{\xi}(c, a) \in \mathbb{R}
    \$

5.  **Q-Function / Critic ($Q_{\psi}$):** Predicts the action-value (expected cumulative future reward) for a state-action pair. An ensemble of $N_q$ critics is used to reduce bias, similar to `REDQ`.
    \$
    q = Q_{\psi}(c, a) \in \mathbb{R}^{N_q}
    \$

6.  **Policy / Actor ($\pi_{\eta}$):** A policy network that maps a latent code to an action. This is used as a prior for the planner and for policy learning.
    \$
    a = \pi_{\eta}(c)
    \$

The following figure (Figure 1 from the original paper) provides a high-level overview of the `DCWM` training process.

![Figure 1: World model training DCWM is a world model with a discrete latent space where each latent state is a discrete code $^ c$ () from a codebook $\\mathcal { C }$ Observations $^ o$ are first mapped through the encoder and then quantized $( \\circledast )$ into one of the discrete codes. We model probabilistic latent transition dynamics $p _ { \\phi } ( \\pmb { c } ^ { \\prime } | \\pmb { c } , \\pmb { a } )$ as a classifier such that it captures a potentially multimodal distribution over the next state $c ^ { \\prime }$ given the previous state $^ c$ and action $^ { a }$ During training, multi-step predictions are made using straight-through (ST) Gumbel-softmax sampling such that gradients backpropagate through time to the encoder. Given this discrete formulation, we train the latent space using a classification objective, i.e. cross-entropy loss. Making the latent representation stochastic and discrete with a codebook contributes to the very high sample efficiency of DC-MPC.](images/1.jpg)
*该图像是示意图，展示了 DCWM（离散码本世界模型）的训练过程。图中显示了从观察 $O_t$ 和 $O_{t+1}$ 经过编码器处理后生成的潜在代码 $c_t$ 和 $c_{t+1}$。利用动态建模 $p_\phi(c_{t+1} | c_t, a_t)$ 进行状态预测，并通过交叉熵损失进行训练。采用 ST Gumbel-softmax 采样方法，使得潜在表示具有随机性和离散性，提高样本效率。*

### 4.2.2. World Model Training
The encoder ($e_{\theta}$), dynamics model ($p_{\phi}$), and reward predictor ($R_{\xi}$) are trained jointly using backpropagation through time (BPTT). The objective function minimizes the sum of a consistency loss for the latent dynamics and a prediction loss for the reward over a horizon of $H$ steps.

The overall objective is:
$$
\mathcal{L}(\theta, \phi, \xi; \mathcal{D}) = \mathbb{E}_{(o, a, o', r)_{0:H} \sim \mathcal{D}} \left[ \sum_{h=0}^{H} \gamma^h \left( \underbrace{\mathrm{CE}\big(p_{\phi}\big(\hat{c}_{h+1} \big| \hat{c}_{h}, a_{h}\big), c_{h+1}\big)}_{\text{Latent-state consistency}} + \underbrace{\big| R_{\xi}\big(\hat{c}_{h}, a_{h}\big) - r_{h} \big| |_2^2}_{\text{Reward prediction}} \right) \right]
$$
with the latent states defined as:
$$
\underbrace{\hat{c}_0 = f(e_{\theta}(o_0))}_{\text{First latent state}} \quad \underbrace{\hat{c}_{h+1} \sim p_{\phi}(\hat{c}_{h+1} | \hat{c}_{h}, a_{h})}_{\text{Stochastic dynamics}} \quad \underbrace{c_h = \mathrm{sg}(f(e_{\theta}(o_h)))}_{\text{Target latent code}}
$$
*   **Data:** The expectation $\mathbb{E}$ is over sequences of observations, actions, and rewards sampled from a replay buffer $\mathcal{D}$.
*   **First Latent State ($\hat{c}_0$):** The initial predicted state for the sequence is obtained by encoding the first observation $o_0$.
*   **Stochastic Dynamics ($\hat{c}_{h+1}$):** Subsequent predicted states are generated by sampling from the dynamics model's output distribution. To allow gradients to flow through this sampling step, the **Gumbel-softmax trick** is used. This is a key detail that enables training the stochastic model.
*   **Target Latent Code ($c_h$):** The "ground truth" or target code for each step is obtained by encoding the *actual* observation $o_h$ from the replay buffer. The stop-gradient operator `sg()` is used to prevent gradients from flowing through the target, ensuring the model learns to predict the target, not just copy it.
*   **Latent-state consistency loss:** This is a **cross-entropy (CE)** loss. It trains the dynamics model $p_{\phi}$ to produce a probability distribution where the highest probability is assigned to the true next latent code $c_{h+1}$.
*   **Reward prediction loss:** This is a standard mean squared error loss that trains the reward predictor $R_{\xi}$.

### 4.2.3. Policy and Value Learning
The policy ($\pi_{\eta}$) and value functions ($Q_{\psi}$) are trained separately from the world model, using an actor-critic method based on `TD3` and `REDQ`. The key difference is that they operate on the discrete latent codes $c$ instead of the raw observations $o$.

**Critic Update:** The ensemble of $N_q=5$ critics is updated by minimizing the TD-error over $N$-step returns.
$$
\mathcal{L}_q(\psi; \mathcal{D}) = \mathbb{E}_{(o, a, o', r)_{n=1}^{N} \sim \mathcal{D}} \left[ \frac{1}{N_q} \sum_{k=1}^{N_q} (q_{\psi_k}(c_t, a_t) - y)^2 \right],
$$
where $c_t = f(e_{\theta}(o_t))$ and the target value $y$ is:
$$
y = \sum_{n=0}^{N-1} \gamma^n r_{t+n} + \gamma^N \min_{k \in \mathcal{M}} q_{\bar{\psi}_k}\big(c_{t+N}, a_{t+N}\big), \quad \text{with } a_{t+n} = \pi_{\bar{\eta}}(c_{t+n}) + \epsilon_{t+n}.
$$
*   $y$ is the target value, composed of the sum of rewards over $N$ steps plus the discounted value of the state at step $t+N$.
*   $\mathcal{M}$ is a set of two randomly subsampled critics from the target network ensemble. Using the minimum of these two helps combat overestimation bias, a core idea from `TD3`.
*   $\bar{\psi}$ and $\bar{\eta}$ denote the parameters of the target networks, which are updated slowly using an exponential moving average.
*   $\epsilon_{t+n}$ is clipped Gaussian noise added to the target policy's actions for smoothing.

    **Actor Update:** The actor is trained to produce actions that maximize the expected Q-value from a random subset of the critics.
$$
\mathcal{L}_{\pi}(\eta; \mathcal{D}) = - \mathbb{E}_{o_t \sim \mathcal{D}} \left[ \frac{1}{|\mathcal{M}|} \sum_{\psi_k \in \mathcal{M}} q_{\psi_k}\big(c_t, \pi_{\eta}(c_t)\big) \right].
$$
*   Here, $c_t = f(e_{\theta}(o_t))$. The actor loss is simply the negative of the average Q-value, so minimizing it corresponds to maximizing the Q-value.

### 4.2.4. Decision-Time Planning
At each step in the environment, `DC-MPC` uses the learned `DCWM` to plan the best action. It uses **Model Predictive Path Integral (MPPI)**, a sampling-based trajectory optimization algorithm. The goal is to find the parameters ($\mu, \sigma$) of a Gaussian distribution over an action sequence of length $H$ that maximizes the expected return.

The objective is:
$$
\mu_{0:H}^*, \sigma_{0:H}^* = \underset{\mu_{0:H}, \sigma_{0:H}}{\mathrm{argmax}} \mathbb{E}_{a_{0:H} \sim \mathcal{N}(\mu_{0:H}, \mathrm{diag}(\sigma_{0:H}^2))} [J(a_{0:H}, o)]
$$
The return of a trajectory $J(a_{0:H}, o)$ is calculated as:
$$
J(a_{0:H}, o) = \sum_{h=0}^{H-1} \gamma^h R_{\xi}(\hat{c}_h, a_h) + \gamma^H \frac{1}{|\mathcal{M}|} \sum_{\psi_k \in \mathcal{M}} q_{\psi_k}(\hat{c}_H, a_H)
$$
subject to the dynamics:
$$
\hat{c}_0 = f(e_{\theta}(o)) \quad \text{and} \quad \hat{c}_{h+1} = \sum_{i=1}^{|\mathcal{C}|} \mathrm{Pr}(\hat{c}_{h+1} = c^{(i)} | \hat{c}_h, a_h) c^{(i)}
$$
*   The trajectory return $J$ is the sum of predicted rewards over the planning horizon $H$, plus a **terminal value** estimated by the learned Q-functions. This "bootstrapping" with the value function allows the planner to account for rewards beyond the finite horizon.
*   **Crucially, during planning, the dynamics are treated deterministically.** Instead of sampling the next state $\hat{c}_{h+1}$, the algorithm calculates the **expected code**, which is a probability-weighted average of all codes in the codebook. This reduces the variance of the planning process, leading to more stable action selection. This is possible because the codebook vectors have a meaningful geometric structure.
*   MPPI iteratively samples action sequences, evaluates them using $J$, and updates the distribution parameters $\mu, \sigma$ based on the best-performing sequences. After a few iterations, the first action of the best sequence is executed in the environment.

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted on three well-established benchmarks for continuous control, featuring a wide range of tasks from locomotion to manipulation.

*   **DeepMind Control Suite (DMControl):** A suite of physics-based simulation tasks (Tassa et al., 2018). The paper uses 30 tasks, including challenging high-dimensional ones like `Humanoid` and `Dog`, which have observation spaces of 67 and 223 dimensions, respectively. These tasks test an agent's ability to learn complex motor skills.
*   **Meta-World:** A benchmark for multi-task and meta-reinforcement learning, focused on simulated robotic manipulation (Yu et al., 2019). The paper uses 45 tasks, such as `Door Open`, `Peg Insert`, and `Button Press`. These tasks evaluate an agent's ability to perform goal-oriented behaviors.
*   **MyoSuite:** A benchmark for musculoskeletal motor control (Vittorio et al., 2022). The paper uses 5 tasks involving a dexterous robotic hand. These are particularly challenging due to the high dimensionality of the action space and complex contact dynamics.

    The choice of these benchmarks is appropriate because they are standard in the field and cover the specific domain of **state-based continuous control** that the paper targets, allowing for direct comparison with prior state-of-the-art methods like `TD-MPC2`.

## 5.2. Evaluation Metrics
The paper uses several metrics to evaluate and compare algorithm performance, aggregated over multiple tasks and random seeds.

### 5.2.1. Episode Return / Success Rate
*   **Conceptual Definition:** This is the most direct measure of performance. **Episode Return** is the sum of all rewards collected by the agent in a single episode (a trial from start to finish). Higher is better. For tasks in Meta-World and MyoSuite, which have binary success criteria, the **Success Rate** (the percentage of successful episodes) is used instead.
*   **Mathematical Formula:**
    \$
    \text{Return} = \sum_{t=0}^{T} \gamma^t r_t
    \$
*   **Symbol Explanation:**
    *   $T$: The length of the episode.
    *   $r_t$: The reward received at time step $t$.
    *   $\gamma$: The discount factor, which prioritizes earlier rewards. For evaluation, it's often set to 1.

### 5.2.2. Interquartile Mean (IQM)
*   **Conceptual Definition:** IQM is a robust statistical metric for aggregating performance across multiple runs or tasks. It calculates the mean of the data after discarding the bottom 25% and top 25% of scores. This makes it less sensitive to extreme outliers than the standard mean, providing a more stable estimate of typical performance.
*   **Mathematical Formula:**
    \$
    \text{IQM}(X) = \frac{2}{n} \sum_{i=n/4+1}^{3n/4} x_{(i)}
    \$
*   **Symbol Explanation:**
    *   $X = \{x_1, x_2, ..., x_n\}$ is the set of $n$ scores (e.g., final returns from $n$ different tasks).
    *   $x_{(i)}$ is the $i$-th score in the sorted list of scores.

### 5.2.3. Optimality Gap
*   **Conceptual Definition:** This metric measures how far an algorithm's performance is from a known optimal or expert level. It is calculated as the average of the normalized scores, where each score is normalized relative to a baseline (random policy) and an expert policy.
*   **Mathematical Formula:**
    \$
    \text{Normalized Score}(x) = \frac{x - x_{\text{random}}}{x_{\text{expert}} - x_{\text{random}}}
    \$
    \$
    \text{Optimality Gap} = 1 - \text{mean}(\text{Normalized Scores})
    \$
*   **Symbol Explanation:**
    *   $x$: The score of the algorithm being evaluated.
    *   $x_{\text{random}}$: The score of a random policy.
    *   $x_{\text{expert}}$: The score of an expert or optimal policy.

        The paper uses these aggregate metrics with stratified bootstrap confidence intervals to ensure statistically robust comparisons, following best practices recommended by Agarwal et al. (2021).

## 5.3. Baselines
The proposed method, `DC-MPC`, is compared against four strong and representative baselines:

*   **DreamerV3:** The state-of-the-art world model that uses a **discrete one-hot** latent space and is trained with observation reconstruction. It serves as the primary baseline for discrete-space models.
*   **TD-MPC2:** The state-of-the-art world model for continuous control that uses a **continuous** latent space and is trained with a consistency loss (decoder-free). It is the main competitor and the top-performing baseline in this domain.
*   **TD-MPC:** The predecessor to `TD-MPC2`, included to show the evolution and performance gains of that model family.
*   **Soft Actor-Critic (SAC):** A top-performing **model-free** RL algorithm. Including SAC helps to establish the sample efficiency benefits of the model-based approaches.

    These baselines are well-chosen as they represent the leading paradigms in both model-based (continuous and discrete latent spaces) and model-free reinforcement learning.

# 6. Results & Analysis

The paper's experiments are designed to answer four key research questions (RQs). The analysis of the results follows this structure.

## 6.1. Core Results Analysis

### 6.1.1. RQ1 & RQ2: Benefits of Discrete, Stochastic, Classification-based Latent Spaces
*   **Question:** Does a discrete latent space offer benefits? Is it the discretization itself, the classification loss, or the stochastic dynamics that helps?
*   **Experiment:** An ablation study (Figure 3) compares different latent space formulations on a subset of DMControl and Meta-World tasks.
    *   `Continuous + MSE` (like TD-MPC2, orange): The baseline continuous model.
    *   `Discrete + MSE` (red): Discrete space but trained with regression.
    *   `Discrete + CE + det`: Discrete space with classification loss but deterministic dynamics.
    *   `Discrete + CE + stoch` (DC-MPC, purple): The full proposed model.

        The following figure (Figure 3 from the original paper) shows the results of this ablation.

        ![Figure 3: Latent space ablation Evaluation of (i) discrete (Discrete) vs continuous (Continuous) latent spaces, (ii) using cross-entropy (CE) vs mean squared error (MSE) for the latent-state consistency loss, and (ii) formulating a deterministic (det) vs stochastic (stoch) dynamics model. Discretizing the latent space (red) improves sample efficiency over the continuous latent space (orange) and formulating stochastic dynamics and training with cross-entropy (purple) improves performance further.](images/3.jpg)
        *该图像是图表，展示了在500,000个环境步骤下的聚合统计数据及训练曲线。左侧统计显示了不同方法的归一化得分，右侧则是DMControl和MetaWorld 10任务的训练曲线，表明离散编码模型在训练效率上优于连续模型。*

*   **Analysis:**
    *   **Discrete vs. Continuous:** The discrete latent space models (red, purple) show better sample efficiency than the continuous one (orange). This supports the claim that discretization is beneficial (C1).
    *   **Classification vs. Regression:** The model trained with classification (purple) outperforms the one trained with regression (red), suggesting that the cross-entropy loss and the modeling of a distribution over next states is important.
    *   **Stochastic vs. Deterministic:** The stochastic model (purple) outperforms its deterministic variant (green), indicating that modeling stochastic dynamics and using Gumbel-softmax sampling during training provides a significant performance boost. This is a key finding, as it suggests that introducing structured noise during training improves representation learning.

### 6.1.2. RQ3: Efficacy of Codebook Encoding
*   **Question:** How does the proposed `codebook` encoding compare to `one-hot` and `label` encodings?
*   **Experiment:** The reward, critic, and policy networks of `DC-MPC` were modified to use different encodings, while the dynamics model always used the codebook.
    *   `Codes` (purple): The full `DC-MPC` model.
    *   `One-hot` (red): Using one-hot vectors.
    *   `Label` (blue): Using integer labels.

        The following figure (Figure 4 from the original paper) presents the results of this comparison.

        ![Figure 4: Discrete encodings ablation DC-MPC with its discrete codebook encoding (purple) outperforms using DC-MPC with one-hot encoding (red) and label encoding (blue), in terms of both sample efficiency (left) and computational efficiency (right). Dynamics model used codes $p _ { \\phi } ( \\mathbf { c } ^ { \\prime } \\mid \\mathbf { c } , \\mathbf { a } )$ whilst reward $R _ { \\xi } ( { \\bf e } , { \\bf a } )$ , critic $\\bar { Q _ { \\psi } } ( { \\bf e } , { \\bf a } )$ and prior policy $\\pi _ { \\eta } ( \\mathbf { e } )$ used the respective encoding e.](images/4.jpg)
        *该图像是一个示意图，展示了不同编码方式在连续控制任务中的表现，包括“Dog Run”和“Humanoid Walk”两种任务。图中比较了具有离散代码本编码（紫色）、一热编码（红色）和标签编码（蓝色）的DC-MPC算法在样本效率和计算效率上的成绩。可以看到，“Codes”方法在各个任务中都表现出明显的优势，特别是在环境步骤和时间的不同场景下，复合编码方式的性能更优。*

*   **Analysis:**
    *   **Codebook is best:** The `codebook` encoding (purple) achieves the best sample efficiency and overall performance.
    *   **Label encoding is poor:** `Label` encoding (blue) performs poorly, especially on the complex `Humanoid Walk` task. This is because it imposes a simplistic 1D ordering on a multi-dimensional state space, losing critical information.
    *   **One-hot is inefficient:** `One-hot` encoding (red) matches the sample efficiency in some tasks but is computationally much slower (right panel of Figure 4). This is because it creates very high-dimensional, sparse inputs for the networks, which is inefficient to process. This confirms the benefits of the dense, low-dimensional codebook representation (C2).

### 6.1.3. RQ4: Comparison with State-of-the-Art
*   **Question:** How does `DC-MPC` compare against leading model-based and model-free algorithms?
*   **Experiment:** `DC-MPC` is benchmarked against `TD-MPC2`, `DreamerV3`, `TD-MPC`, and `SAC` on the full suites of DMControl, Meta-World, and MyoSuite.

    The following figure (Figure 5 from the original paper) shows the aggregate training curves across all three benchmarks.

    ![Figure 5: Aggregate training curves in DMControl, Meta-World, & MyoSuite DC-MPC generally matches TD-MPC2 whilst outperforming DreamerV3, SAC and TD-MPC across all tasks. We plot the mean (solid line) and the $9 \\hat { 5 } \\%$ confidence intervals (shaded) across 3 seeds per task.](images/5.jpg)
    *该图像是图表，展示了在 DMControl、Meta-World 和 MyoSuite 三个任务集上，不同算法的训练曲线。DC-MPC 的表现与 TD-MPC2 相当，且在所有任务上优于 DreamerV3、SAC 和 TD-MPC。图中展示了均值（实线）和 $95\\%$ 的置信区间（阴影部分）。*

*   **Analysis:**
    *   **Overall Performance:** `DC-MPC` performs competitively with the state-of-the-art `TD-MPC2`, generally matching its performance across the board. Both significantly outperform `DreamerV3`, `TD-MPC`, and `SAC`.
    *   **High-Dimensional Tasks:** `DC-MPC` particularly excels in the most complex, high-dimensional DMControl tasks like `Dog` and `Humanoid` (Figure 13). In these tasks, it significantly outperforms all baselines, including `TD-MPC2`. The authors hypothesize that the discretization is especially beneficial for simplifying the learning of dynamics in these very large state spaces.
    *   **Manipulation Tasks:** In Meta-World and MyoSuite, `DC-MPC` again matches the performance of `TD-MPC2` and substantially outperforms `DreamerV3`.

        The following figure (Figure 13 from the original paper) highlights the strong performance in high-dimensional tasks.

        ![Figure 13: High-dimensional locomotion DC-MPC (purple) significantly outperforms TD-MPC2 (blue) and DreamerV3 (red) in the complex, high-dimensional locomotion tasks from DMControl.](images/13.jpg)
        *该图像是图表，展示了在复杂高维运动任务中，DC-MPC（紫色）在不同环境步骤下相比于TD-MPC2（蓝色）和DreamerV3（红色）的表现，DC-MPC显著优于其他算法。*

    The following tables (transcribed from Figures 14, 16, and 18) summarize the aggregate performance at 1M steps.

    The following are the results from Figure 14 of the original paper:

    ![Figure 14: DMControl aggregate results DC-MPC generally outperforms TD-MPC2 and DreamerV3 in DMControl tasks. This is due to DC-MPC's strong performance in the hard Dog and Humanoid tasks. Error bars represent $9 5 \\%$ stratified bootstrap confidence intervals.](images/14.jpg)
    *该图像是图表，展示了在 DMControl 环境中不同算法（如 TD-MPC、DreamerV3、TD-MPC2 和 DC-MPC）在 30 个任务中的表现。图中以中位数、IQM、均值和最优性差距展示了各算法的归一化收益，DC-MPC 在多个指标上表现优异。*

    The following are the results from Figure 16 of the original paper:

    ![Figure 16: Meta-World results DC-MPC performs well in Meta-World, generally matching TDMPC2, whilst significantly outperforming DreamerV3 and SAC. Error bars represent $9 5 \\%$ stratified bootstrap confidence intervals.](images/16.jpg)
    *该图像是图表，展示了在 MetaWorld 环境中不同算法的表现，包括 TD-MPC、SAC、DreamerV3、TD-MPC2 和 DC-MPC。图中显示了在 1M 步长下，算法的成功率、IQM、均值和最优性差距等指标的比较。*

    The following are the results from Figure 18 of the original paper:

    ![Figure 18: MyoSuite results DC-MPC performs similarly to TD-MPC2 in MyoSuite. Error bars represent $9 5 \\%$ stratified bootstrap confidence intervals.](images/18.jpg)
    *该图像是一个图表，展示了在MyoSuite中的结果。DC-MPC在多个任务中的表现与TD-MPC2相似，图中显示不同方法的成功率，包括中位数、IQM、平均值和最优性差距等指标。*

    These results robustly validate the paper's main claim (C3): `DC-MPC` is a highly competitive algorithm for continuous control.

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Impact of Integrating DCWM into Other Models
*   **DCWM + TD-MPC2:** When the discrete, stochastic latent space of `DCWM` is integrated into the `TD-MPC2` architecture, it improves performance over the original `TD-MPC2` (Figure 6). This is strong evidence that the latent space design itself is a key contributor to performance, independent of the other architectural choices in `DC-MPC`.
    The following figure (Figure 6 from the original paper) shows this result.

    ![Figure 6: TD-MPC2 with DCWM Adding DC-MPC's discrete and stochastic latent space to TD-MPC2 improves performance. See Apps. B and B.10 for more details.](images/6.jpg)
    *该图像是图表，展示了在1M环境步数下TD-MPC2与DCWM结合后的性能表现与DC-MPC的对比。左侧为聚合统计，右侧显示DMControl和Meta-World的训练曲线。结果表明，将DCWM引入TD-MPC2显著提升了性能。*

*   **DCWM + DreamerV3:** In contrast, adding the `DCWM` codebook to `DreamerV3` does not help (Figure 19). The authors show in a further experiment (Figure 20) that `DreamerV3`'s reliance on **observation reconstruction** is the likely culprit. Adding a reconstruction loss to `DC-MPC` severely harms its performance, especially in complex tasks. This reinforces the argument that for state-based control, decoder-free, consistency-based objectives are superior.

### 6.2.2. Sensitivity to Hyperparameters
*   **Codebook Size ($|\mathcal{C}|$):** The performance of `DC-MPC` is not overly sensitive to the codebook size (Figure 7). A size of 15 ($L={5,3}$) works well across tasks. As expected, smaller codebooks are "activated" (all codes are used) faster during training.
*   **Latent Dimension ($d$):** The algorithm is also robust to the latent dimension (Figure 8), although a dimension that is too small can harm performance on more difficult tasks like `Humanoid Walk`. A default of $d=512$ is effective.

### 6.2.3. Quantization Method (FSQ vs. VQ)
*   The paper compares FSQ to the more common Vector Quantization (VQ) method (Figure 10). FSQ performs on par with or better than VQ, while being simpler as it does not require learning the codebook vectors or using extra loss terms, which stabilizes training.

### 6.2.4. Ensemble of Critics (REDQ)
*   The use of an ensemble of $N_q=5$ critics (from `REDQ`) provides a slight benefit over the standard double Q-learning approach ($N_q=2$) in the hardest tasks (Figure 11), justifying its inclusion.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that discrete latent spaces, when designed correctly, are highly effective for state-based continuous control. The authors introduce **DC-MPC**, a model-based RL algorithm built upon a **Discrete Codebook World Model (DCWM)**. The key findings are:
1.  A discrete latent space learned with a **classification objective** (cross-entropy loss) on latent state transitions is more sample-efficient than a continuous space learned with regression.
2.  Modeling the latent dynamics as **stochastic** and using Gumbel-softmax sampling during training is a crucial element for high performance.
3.  A **codebook encoding** (specifically using FSQ) is a superior representation for discretizing continuous states compared to `one-hot` or `label` encodings, as it preserves ordinal structure while being dense and efficient.
4.  The resulting `DC-MPC` algorithm achieves state-of-the-art performance, matching or exceeding strong baselines like `TD-MPC2` and `DreamerV3` on challenging continuous control benchmarks.

    These results open a promising research direction for utilizing discrete representations in domains previously dominated by continuous models.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations and suggest future research avenues:
*   **Hyperparameter Tuning:** `DC-MPC` still requires some task-specific tuning, particularly for the exploration noise schedule and the $N$-step return length. A more robust, hyperparameter-free version would be a valuable extension.
*   **Principled Exploration:** The current exploration mechanism relies on adding scheduled noise. A more advanced approach would be to model the world model's epistemic uncertainty (its confidence in its own predictions) and use that to guide exploration, potentially removing the need for manual tuning.
*   **Scaling and Generalization:** The authors suggest investigating how `DC-MPC` scales with larger models and more data, and whether its discrete representation is suitable for building generalist, multi-task agents, which is a major trend in modern AI research.
*   **Alternative Backbones:** The current model uses MLPs. Future work could explore more powerful architectures like Transformers or diffusion models as the backbone for the world model.

## 7.3. Personal Insights & Critique
This is a strong, well-executed paper that makes a clear and empirically-backed argument.
*   **Key Insight:** The most significant contribution is the successful synthesis of ideas from two different lines of world model research. It takes the discrete latent space concept from the `Dreamer` line and the decoder-free, consistency-based learning from the `TD-MPC` line, and shows that the combination, with the right encoding (codebook), is more powerful than either approach in isolation for this specific problem domain.
*   **Stochasticity for Deterministic Environments:** A fascinating result is that modeling dynamics as *stochastic* during training improves performance even in *deterministic* simulation environments. This suggests that the stochasticity acts as a powerful regularizer, forcing the model to learn more robust and generalizable representations by not overfitting to a single deterministic outcome. The use of Gumbel-softmax is key to making this tractable.
*   **Potential Issues/Critique:**
    *   The "expected code" trick used during planning feels somewhat heuristic. While it works well empirically, it means the planner operates on states (weighted averages of codes) that are not actually part of the discrete codebook. This breaks the discrete abstraction and could potentially cause issues in scenarios where interpolation is not meaningful. A deeper theoretical analysis of why this works would be beneficial.
    *   The paper focuses exclusively on state-based inputs. While this is its stated goal, the real challenge for world models often lies in learning from high-dimensional observations like images. It remains an open question whether the benefits of this specific codebook approach would translate to visual control, or if the reconstruction-based methods like `DreamerV3` would still hold an advantage there.
*   **Inspiration for Other Domains:** The success of FSQ and codebook representations in this RL context could inspire similar applications in other sequence modeling tasks where an underlying continuous process is being modeled, such as time-series forecasting, physics simulation, or even language modeling (where discrete tokens represent a continuous semantic space). The idea of a structured, discrete representation as a bridge between continuous data and discrete processing is very powerful.