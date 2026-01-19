# 1. Bibliographic Information

## 1.1. Title
Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning

## 1.2. Authors
Qi Wang, Zhipeng Zhang, Baao Xie, Xin Jin, Yunbo Wang, Shiyu Wang, Liaomo Zheng, Xiaokang Yang, and Wenjun Zeng.

The authors are affiliated with several prominent institutions in China, including Shanghai Jiao Tong University, the Eastern Institute of Technology in Ningbo, the University of Chinese Academy of Sciences, and the Shenyang Institute of Computing Technology. This indicates a collaborative research effort between multiple academic and research organizations.

## 1.3. Journal/Conference
The paper is listed as a preprint on arXiv, submitted on March 11, 2025 (a future date, which is a common practice for papers under review for major conferences). Given the topic and authors' publication history, it is likely intended for a top-tier machine learning or computer vision conference such as ICLR, NeurIPS, or ICML. These venues are highly competitive and influential in the field of AI.

## 1.4. Publication Year
The publication date on arXiv is given as 2025.

## 1.5. Abstract
The abstract introduces a key challenge in visual reinforcement learning (RL): poor sample efficiency in environments with visual variations. While disentangled representation learning has been proposed to address this, existing methods typically learn from scratch. This paper proposes a novel approach, **Disentangled World Models (DisWM)**, which instead learns semantic knowledge from "distracting videos" (videos with visual variations) and transfers it to the RL agent. The method operates in two phases:
1.  **Offline Pretraining:** An action-free video prediction model is pretrained on distracting videos using disentanglement regularization to learn to separate semantic factors.
2.  **Online Finetuning:** This disentanglement capability is transferred to a world model via "offline-to-online latent distillation." The world model is then finetuned in the target RL environment, using actions and rewards to further enhance the disentangled representations.
    The authors state that their experimental results on various benchmarks validate the superiority of this approach.

## 1.6. Original Source Link
-   **Original Source Link:** https://arxiv.org/abs/2503.08751
-   **PDF Link:** https://arxiv.org/pdf/2503.08751v2
-   **Publication Status:** This is a preprint available on arXiv. It has not yet been peer-reviewed or officially published in a conference or journal.

# 2. Executive Summary

## 2.1. Background & Motivation
-   **Core Problem:** Visual Reinforcement Learning (VRL) agents, which learn to make decisions from raw pixel inputs (like images or videos), struggle to adapt to new situations. Even minor, semantically irrelevant changes in the environment—such as a change in lighting, background color, or object texture—can drastically alter the pixel values of observations. This causes the agent's learned policy to fail, a problem known as poor **generalization** and low **sample efficiency** (requiring vast amounts of experience to learn).
-   **Existing Gaps:** Many prior methods try to solve this by learning **disentangled representations**, where different underlying factors of the world (e.g., object position, color, size) are captured by separate, independent dimensions in the learned feature space. However, these methods usually start learning from a blank slate (`from scratch`) within the target RL environment. This process is inefficient, as the agent must simultaneously learn the task's objective and how to ignore visual distractions.
-   **Innovative Idea:** This paper's central idea is to decouple the learning of visual semantics from the learning of the control task. The authors propose to first learn a general "understanding" of visual variations from a large, readily available source of **distracting videos**. These videos do not need to be from the exact same environment as the final task and do not require action or reward labels. The knowledge of how to disentangle visual factors is then **transferred** to the RL agent, which can then focus its limited environmental interactions on learning the actual task, leading to much faster learning.

## 2.2. Main Contributions / Findings
The paper makes the following key contributions:
1.  **A Novel Problem Formulation:** It frames the challenge of robust VRL as a **domain transfer learning problem**. The goal is to transfer a "disentanglement capability" learned from a source domain (distracting videos) to a target domain (the RL task).
2.  **The Disentangled World Models (DisWM) Framework:** This is a comprehensive model-based RL framework that implements the transfer learning idea. It operates in a pretraining-finetuning paradigm and consists of three key technical components:
    *   **Disentangled Representation Pretraining:** Uses a `β-VAE`-based video model to learn to disentangle semantic factors from action-free videos offline.
    *   **Offline-to-Online Latent Distillation:** A novel technique to transfer the learned disentanglement knowledge from the pretrained model (teacher) to the world model (student) by aligning their latent distributions. This avoids "catastrophic forgetting" of the disentangled features during finetuning.
    *   **Disentangled World Model Adaptation:** The world model is finetuned online with the RL task, incorporating actions and rewards. It is guided by a flexible disentanglement constraint, and the new interactions from the agent further enrich the data, strengthening the disentanglement process.
3.  **Superior Performance:** The findings show that DisWM significantly outperforms state-of-the-art VRL baselines in terms of sample efficiency and final performance on several challenging benchmarks with visual distractions. This demonstrates that pre-learning semantic representations from videos is a highly effective strategy for improving VRL agent robustness.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Reinforcement Learning (RL)
Reinforcement Learning is a paradigm of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The process is typically modeled as follows:
-   At each timestep, the agent observes the environment's **state** (or an **observation** of the state).
-   Based on this observation, the agent chooses an **action**.
-   The environment responds by transitioning to a new state and providing the agent with a numerical **reward**.
-   The agent's goal is to learn a **policy**—a mapping from states to actions—that maximizes the cumulative reward over time.

### 3.1.2. Partially Observable Markov Decision Process (POMDP)
In many real-world scenarios, the agent does not have access to the true, complete state of the environment. For example, in visual RL, a single image from a camera is just a partial snapshot and doesn't reveal hidden information like object velocities or internal system states. A POMDP formalizes this by distinguishing between the environment's true state $s$ and the **observation** $o$ that the agent receives. A POMDP is defined by the tuple $⟨O, A, T, R, γ⟩$, where $O$ is the space of observations, $A$ is the action space, $T$ is the transition probability, $R$ is the reward function, and $γ$ is the discount factor.

### 3.1.3. Model-Based Reinforcement Learning (MBRL)
RL methods can be broadly categorized into two types:
-   **Model-Free RL:** The agent directly learns a policy ($π(a|s)$) or a value function (`Q(s, a)`) without explicitly learning the environment's dynamics. It's like learning through trial and error.
-   **Model-Based RL (MBRL):** The agent first learns a **world model**, which is a simulation of the environment. This model predicts the next state and reward given the current state and an action ($s_t+1, r_t+1 = M(s_t, a_t)$). Once the model is learned, the agent can "imagine" or "dream" future trajectories within its learned model to plan and learn a policy, which is often much more sample-efficient than interacting with the real world. DisWM is an MBRL method.

### 3.1.4. Disentangled Representation Learning (DRL)
The goal of DRL is to learn a data representation where different, meaningful factors of variation in the data are separated and encoded into distinct, independent dimensions of a latent vector $z$. For example, in an image of a face, one latent dimension might control the smile, another the head pose, and a third the lighting direction. This makes the representation interpretable and robust, as changing one real-world factor only affects a small part of the latent code.

### 3.1.5. Variational Autoencoder (VAE) and β-VAE
A **Variational Autoencoder (VAE)** is a generative model that learns a compressed latent representation of data. It has two main parts:
-   An **encoder** network that maps an input $x$ to a probability distribution in the latent space (typically a Gaussian with mean $μ$ and variance $σ²$).
-   A **decoder** network that takes a sample $z$ from this latent distribution and tries to reconstruct the original input $x$.
    The VAE is trained to minimize a loss function composed of two terms: a **reconstruction loss** (how well the decoder reconstructs the input) and a **Kullback-Leibler (KL) divergence** loss. The KL term forces the learned latent distributions to be close to a standard prior (e.g., a standard normal distribution $N(0, I)$), which regularizes the latent space.

The **β-VAE** is a modification of the VAE that introduces a hyperparameter $β$ to scale the KL divergence term.
\$
\mathcal{L}_{VAE} = \mathbb{E}[\log p(x|z)] - \beta \cdot \mathrm{KL}(q(z|x) \| p(z))
\$
When $β > 1$, it puts stronger pressure on the model to make the latent dimensions independent, thus encouraging better **disentanglement**. The DisWM framework heavily relies on this principle.

### 3.1.6. Knowledge Distillation
Knowledge distillation is a technique for transferring knowledge from a large, powerful "teacher" model to a smaller, more efficient "student" model. This is often done by training the student to mimic the teacher's outputs. For example, the student's loss function might include a term that minimizes the difference between its output probability distribution and the teacher's. In this paper, distillation is used in the latent space: the world model (student) is trained to produce latent representations that match the distributions of the highly disentangled representations from the pretrained video model (teacher).

## 3.2. Previous Works
The paper positions itself relative to several key areas of research:
-   **Visual MBRL (e.g., DreamerV2):** `DreamerV2` [10] is a highly influential MBRL framework that learns a world model from pixels and uses it to train an actor-critic agent entirely in imagined trajectories. DisWM builds on this world model paradigm but augments it with pretraining and disentanglement.
-   **Transfer RL from Videos (e.g., APV):** `APV` [29] also proposes a pretraining-finetuning approach for VRL using action-free videos. It learns representations via pretraining and then finetunes on the downstream task. DisWM differentiates itself by focusing specifically on *distracting* videos and using explicit disentanglement constraints and a novel latent distillation mechanism for more effective knowledge transfer.
-   **Disentanglement in RL (e.g., TED):** `TED` [7] is a VRL method designed to handle visual distractions by learning temporally disentangled representations through a self-supervised auxiliary task. Unlike DisWM, `TED` learns from scratch on the target task and does not leverage external video data.
-   **Contrastive Learning in RL (e.g., CURL):** `CURL` [18] is a model-free method that uses contrastive learning to learn useful representations from augmented views of observations, improving sample efficiency. It is a different approach to representation learning compared to the generative, disentanglement-focused method of DisWM.

## 3.3. Technological Evolution
The field of VRL has evolved from methods that were brittle to visual changes to more robust approaches:
1.  **Early VRL:** Directly applied RL algorithms to image inputs, often suffering from low sample efficiency and poor generalization.
2.  **Representation Learning for VRL:** Researchers recognized the need for better representations. This led to methods incorporating auxiliary losses, data augmentation (`RAD`), or contrastive learning (`CURL`) to learn more invariant features.
3.  **Disentanglement for VRL:** A more structured approach to representation learning emerged, with methods like `TED` attempting to explicitly disentangle causal factors of the environment to improve generalization to visual variations. These methods, however, still learned from scratch.
4.  **Pretraining & Transfer for VRL:** The latest wave of research, which includes `APV` and this paper (DisWM), focuses on leveraging large, unlabeled video datasets to pre-train powerful representations before tackling the specific RL task. This is inspired by the success of pretraining in NLP (e.g., BERT) and computer vision (e.g., ImageNet pretraining). DisWM sits at this frontier, uniquely combining pretraining, disentanglement, and world models.

## 3.4. Differentiation Analysis
Compared to the most related works, DisWM's core innovations are:
-   **vs. APV:** While both use video pretraining, DisWM's pretraining is explicitly focused on **disentanglement** using a `β-VAE` objective. Furthermore, its knowledge transfer mechanism, **offline-to-online latent distillation**, is designed to be more robust to domain shifts than the simple finetuning or intrinsic bonus approach used in `APV`.
-   **vs. TED:** `TED` learns disentangled representations online, concurrently with the RL task. DisWM separates these concerns: it first learns a powerful disentangled representation offline from diverse videos, and then transfers this capability. This pre-computation of semantic knowledge is hypothesized to be more sample-efficient.
-   **vs. DreamerV2:** DisWM augments the powerful `DreamerV2` world model architecture with the ability to leverage prior knowledge from external videos and an explicit mechanism to handle visual distractions through disentanglement, which the original `DreamerV2` lacks.

# 4. Methodology
The proposed method, Disentangled World Models (DisWM), is a model-based RL framework that follows a pretraining-finetuning paradigm. It consists of three main stages, as detailed in Figure 2 of the paper.

The overall workflow is outlined in Algorithm 1.

![该图像是示意图，展示了无动作的解耦表示预训练和动作条件的世界模型微调过程。左侧（a）展示了通过 `eta`-VAE 进行解耦表示学习，右侧（b）则显示了如何通过潜在蒸馏将知识转移至世界模型，并在在线环境中进行微调。](images/2.jpg)
*该图像是示意图，展示了无动作的解耦表示预训练和动作条件的世界模型微调过程。左侧（a）展示了通过 `eta`-VAE 进行解耦表示学习，右侧（b）则显示了如何通过潜在蒸馏将知识转移至世界模型，并在在线环境中进行微调。*

## Algorithm 1: The training pipeline of DisWM.
1.  **Hyperparameters:** Initialize hyperparameters, including $H$, the horizon for latent imagination.
2.  **Initialize:** Initialize model parameters (`ϕ`, $ψ$, $ξ$) and load the distracting video dataset $D$.
3.  **Phase 1: Disentangled Representation Pretraining (Offline)**
    *   For a number of steps (`K₁`), sample minibatches of observations ${o_t}$ from the distracting video dataset $D$.
    *   Pretrain an action-free video prediction model using a disentanglement objective (Equation 2).
4.  **Initialize Replay Buffer:** A random agent interacts with the environment to collect an initial set of experiences in a replay buffer $B$.
5.  **Phase 2 & 3: Online Adaptation Loop**
    *   The main training loop begins.
    *   For a number of steps (`K₂`):
        *   Sample batches of transitions $(o_t, a_t, r_t)$ from the replay buffer $B$.
        *   **Offline-to-Online Latent Distillation:** Distill knowledge from the pretrained model to the world model using Equation (3).
        *   **Disentangled World Model Adaptation:** Train the main world model `M_ϕ` using a combined loss (Equation 5) that includes reconstruction, prediction, disentanglement, and distillation terms.
        *   **Behavior Learning:** Use the trained world model `M_ϕ` to generate imagined trajectories. Train the actor ($π_ψ$) and critic ($v_ξ$) on these imagined trajectories.
    *   **Environment Interaction:** Use the newly trained actor $π_ψ$ to interact with the real environment for $T$ timesteps, collecting new data $(o_t, a_t, r_t)$.
    *   Add this new data to the replay buffer $B$.
    *   Repeat the online adaptation loop until convergence.

## 4.1. Stage 1: Disentangled Representation Pretraining
The first stage aims to learn a "knowledgeable" encoder that can extract disentangled semantic features from images. This is done offline using a dataset of **distracting videos**, which are videos from related (but not necessarily identical) domains that contain visual variations like changing colors or backgrounds. This model is action-free.

## 4.1.1. Model Architecture
The pretrained model is a video prediction model based on a `β-VAE`. Its components are:
-   **`β-VAE` encoder:** Encodes an observation $o_t$ into a latent representation. The paper denotes this as $z_t = e_φ'(o_t)$ for the encoded features, which are then used to parameterize the posterior distribution.
-   **Posterior state:** $z_t ~ q_φ'(z_t | z_t-1, o_t)$. This is the latent state inferred from both the current observation $o_t$ and the previous latent state $z_t-1$.
-   **Prior state:** $hat{z}_t ~ p_φ'(hat{z}_t | z_t-1)$. This is a prediction of the next latent state based only on the previous latent state. The model learns dynamics in the latent space.
-   **Decoder:** $hat{o}_t ~ p_φ'(hat{o}_t | z_t)$. Reconstructs the original observation from the latent state $z_t$.

## 4.1.2. Pretraining Loss Function
The model is trained to minimize the following loss function, as given in Equation (2):
\$
\mathcal{L}(\phi') = \mathbb{E}_{q_{\phi'}} \left[ \sum_{t=1}^{T} \underbrace{-\ln p_{\phi'}(o_t | z_t)}_{\text{image reconstruction}} + \underbrace{\beta_1 \mathrm{KL}[q_{\phi'}(\boldsymbol{z}_t | \boldsymbol{z}_{t-1}, \boldsymbol{o}_t) \| p_{\phi'}(\hat{\boldsymbol{z}}_t | \boldsymbol{z}_{t-1})]}_{\text{action-free KL loss}} + \underbrace{\beta_2 \mathrm{KL}[q_{\phi'}(\mathbf{z}_t | \boldsymbol{o}_t) \| p(\mathbf{z}_t)]}_{\text{disentanglement loss}} \right]
\$
-   **`image reconstruction` term:** This is the standard VAE reconstruction loss. It ensures that the latent state $z_t$ contains enough information to reconstruct the original observation $o_t$.
-   **`action-free KL loss` term:** This term, weighted by $β₁$, trains the latent dynamics model. It forces the posterior distribution $q$ (which sees the observation) to match the prior distribution $p$ (which only sees the history). This encourages the model to predict future latent states accurately.
-   **`disentanglement loss` term:** This is the core `β-VAE` term, weighted by $β₂$. It forces the aggregated posterior distribution over the entire dataset, $q(z_t | o_t)$, to be close to a standard multivariate Gaussian prior $p(z_t) = N(0, I)$. This pressure encourages the individual dimensions of the latent vector $z_t$ to become independent and thus disentangled.

    The output of this stage is a "teacher" model with parameters $φ'$, which has learned to produce a well-disentangled latent variable, here denoted as `z_disen`.

## 4.2. Stage 2: Offline-to-Online Latent Distillation
After pretraining, the agent begins to learn the downstream RL task online. A naive approach would be to initialize the new world model with the pretrained weights $φ'$. However, due to domain shifts (e.g., different dynamics, appearances), finetuning can quickly destroy the learned disentanglement ("catastrophic forgetting").

To prevent this, the paper introduces a **latent distillation** loss. The goal is to transfer the *disentanglement capability* from the pretrained model (teacher) to the new world model being trained for the task (student).

The distillation loss is the KL divergence between the latent distributions produced by the teacher and student models for the same observation. Let `z_disen` be the latent variable from the fixed, pretrained teacher model, and `z_task` be the latent variable from the world model being trained. The distillation loss is given by Equation (3):
\$
\mathcal{L}_{\mathrm{distill}} = \mathrm{KL}(\mathbf{z}_{\mathrm{disen}} \| \mathbf{z}_{\mathrm{task}}) = \sum \mathbf{z}_{\mathrm{disen}} \cdot \log\left( \frac{\mathbf{z}_{\mathrm{disen}}}{\mathbf{z}_{\mathrm{task}}} \right)
\$
-   This loss encourages the student model's latent variable `z_task` to have a distribution that matches the teacher's `z_disen`. Since `z_disen` is highly disentangled, this effectively teaches the student model to also produce disentangled representations.

## 4.3. Stage 3: Disentangled World Model Adaptation
In this final stage, a full world model `M_ϕ` is trained online using data from the agent's interactions with the target environment. This model is architecturally similar to that in DreamerV2 but incorporates the new disentanglement and distillation constraints.

## 4.3.1. World Model Architecture
The world model `M_ϕ` has the following components:
-   **Recurrent transition model:** $h_t = f_ϕ(h_t-1, z_t-1, a_t-1)$. This is an RNN that summarizes history, taking the previous hidden state $h_t-1$, latent state $z_t-1$, and action $a_t-1$ to produce the current hidden state $h_t$.
-   **Encoder and State Models:** Similar to the pretraining model, but now conditioned on the history $h_t$:
    -   Posterior state: $z_t ~ q_ϕ(z_t | h_t, o_t)$
    -   Prior state: $tilde{z}_t ~ p_ϕ(tilde{z}_t | h_t)$
-   **Prediction Heads:**
    -   Reconstruction: $hat{o}_t ~ p_ϕ(hat{o}_t | h_t, z_t)$
    -   Reward prediction: $hat{r}_t ~ r_ϕ(hat{r}_t | h_t, z_t)$
    -   Discount factor prediction: $hat{γ}_t ~ p_ϕ(hat{γ}_t | h_t, z_t)$ (predicts if the episode terminates).

## 4.3.2. Adaptation Loss Function
The world model `M_ϕ` is trained with a comprehensive loss function, given in Equation (5):
\$
\begin{array}{rcl}
\mathcal{L}(\phi) & = & \mathbb{E}_{q_{\phi}} \Big[ \sum_{t=1}^{T} \underbrace{-\ln p_{\phi}(o_t | h_t, z_t)}_{\text{img reconstruction}} \underbrace{-\ln r_{\phi}(r_t | h_t, z_t)}_{\text{reward prediction}} \\
& & \underbrace{-\ln p_{\phi}(\gamma_t | h_t, z_t)}_{\text{discount prediction}} \underbrace{+ \alpha \mathrm{KL}[q_{\phi}(z_t | h_t, o_t) \| p_{\phi}(\hat{z}_t | h_t)]}_{\text{KL divergence}} \\
& & \underbrace{+ \beta \mathrm{KL}[q_{\phi}(\mathbf{z}_t | o_t) \| p(\mathbf{z}_t)]}_{\text{disentanglement}} + \underbrace{\eta \mathcal{L}_{\mathrm{distill}}}_{\text{distillation}} \Big]
\end{array}
\$
-   **Prediction Terms:** The first three terms are standard in world models like Dreamer. They train the model to reconstruct observations and predict rewards and discounts.
-   **`KL divergence` term:** Weighted by $α$, this is the standard dynamics learning objective, forcing the posterior $q$ to match the prior $p$, making the model's predictions of the future consistent with reality.
-   **`disentanglement` term:** Weighted by $β$, this is the same `β-VAE` disentanglement loss from the pretraining stage, now applied during online adaptation. It ensures the model continues to learn and maintain disentangled representations as it interacts with the new environment.
-   **`distillation` term:** Weighted by $η$, this is the latent distillation loss from Equation (3). It continuously transfers the disentanglement knowledge from the pretrained teacher. The weight $η$ is gradually decreased during training, suggesting that the direct guidance from the teacher is more important early on, and the model becomes more self-reliant later.

    Finally, a standard actor-critic method (identical to `DreamerV2`) is used to learn a policy by planning within the learned world model `M_ϕ`.

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted on several visual reinforcement learning benchmarks to test the method's effectiveness in handling visual distractions.
-   **DeepMind Control Suite (DMC):** A popular set of continuous control tasks (e.g., `Walker Walk`, `Cheetah Run`, `Hopper Stand`, `Finger Spin`, `Cartpole Swingup`). The authors modify these environments by adding visual distractors, specifically by changing the colors of objects and backgrounds during training.
-   **MuJoCo Pusher:** A robotic manipulation task where a multi-jointed arm must push a cylinder to a target location. This is also modified with color distractors.
-   **DrawerWorld:** A manipulation task based on MetaWorld, designed to test adaptability to texture changes. The agent is trained with a grid texture, then a wood texture, and evaluated on an unseen metal texture.
-   **Distracting Video Datasets:** For pretraining, the authors generate datasets of 1 million frames by running a `DreamerV2` agent in DMC environments with color distractors and collecting the observations. An important aspect is the **cross-domain** setup, where videos from one task (e.g., DMC Reacher) are used to pretrain an agent for a different task (e.g., MuJoCo Pusher).

    The following figure from the paper shows examples of the modified environments with color distractors.

    ![Figure 3. Example image observations of our modified DMC and MuJoCo Pusher with color distractors.](images/3.jpg)
    *该图像是示意图，展示了修改后的 DMC 和 MuJoCo Pusher 环境中的示例图像观察，包含不同颜色的干扰物体。上方展示的是 'Walker Walk' 任务，中间为 'Reacher Easy' 任务，底部则是 'Pusher' 任务。*

The paper also highlights the significant domain gap in a cross-domain experiment, summarized in Table 1.

The following are the results from Table 1 of the original paper:

| | Video: DMC | Target: MuJoCo | Similarity / Difference |
| :--- | :--- | :--- | :--- |
| **Task** | Reacher Easy | Pusher | Relevant robotic control tasks |
| **Dynamics** | Two-link planar | Multi-jointed robot arm | Different |
| **Action space** | Box(-1, 1, (2,), float32) | Box(-2, 2, (7,), float32) | Different |
| **Reward range** | [0, 1] | [-4.49, 0] | Different |

This table clearly shows that the pretraining videos and the downstream task can differ in dynamics, action space, and reward structure, making the knowledge transfer non-trivial.

## 5.2. Evaluation Metrics
The primary metrics used to evaluate the performance of the RL agents are:
-   **Episode Return:**
    1.  **Conceptual Definition:** This metric measures the total accumulated reward an agent receives over a single episode (a single attempt at the task, from start to finish). A higher episode return indicates better performance, as the agent is successfully executing behaviors that the reward function is designed to encourage. It is the most common metric for evaluating RL agent performance.
    2.  **Mathematical Formula:** For an episode of length $T$, the undiscounted episode return $R$ is calculated as:
        \$
        R = \sum_{t=0}^{T-1} r_t
        \$
    3.  **Symbol Explanation:**
        *   $r_t$: The reward received at timestep $t$.
-   **Success Rate (%):**
    1.  **Conceptual Definition:** This metric is used for tasks with a clear binary outcome (success or failure), such as reaching a goal or opening a drawer. It is defined as the percentage of episodes in which the agent successfully completes the task's objective.
    2.  **Mathematical Formula:**
        \$
        \text{Success Rate} = \frac{\text{Number of Successful Episodes}}{\text{Total Number of Evaluation Episodes}} \times 100\%
        \$
    3.  **Symbol Explanation:** N/A.

## 5.3. Baselines
The proposed method, DisWM, is compared against several strong baselines representing different approaches to VRL:
-   **`DreamerV2`:** A state-of-the-art model-based RL agent that learns from scratch without any pretraining or explicit disentanglement mechanisms.
-   **`APV`:** A transfer learning baseline that also uses action-free video pretraining, serving as a close competitor to DisWM.
-   **`DV2 Finetune`:** A simple transfer learning baseline where a full `DreamerV2` model is pretrained on distracting videos and then finetuned on the target task. This helps isolate the benefit of DisWM's specific distillation strategy.
-   **`TED`:** A baseline that learns disentangled representations online to cope with distractors but does not use pretraining.
-   **`CURL`:** A model-free RL method that uses contrastive learning for representation learning, representing a different family of algorithms.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main results, presented in Figure 4, show the learning curves (episode return vs. environment steps) for DisWM and the baselines across several DMC tasks and the cross-domain MuJoCo task.

![Figure 4. Comparison of DisWM against visual RL baselines, including DreamerV2 \[10\], `A P V` \[29\], DV2 Finetune, TED \[7\], CURL \[18\].](images/4.jpg)
*该图像是图表，展示了DisWM与多个视觉强化学习基线（如DreamerV2、APV等）的比较。图中显示在不同环境步骤下各算法的表现，包括每个算法的回报率变化情况。*

-   **Superior Sample Efficiency:** Across all tasks, DisWM (the red curve) consistently achieves higher returns with fewer environment steps compared to all baselines. This demonstrates its superior sample efficiency. The agent learns the desired behavior much faster.
-   **Effectiveness against Distractors:** The performance gap is particularly noticeable compared to `DreamerV2`, which struggles in these visually distracting environments. This confirms that explicitly handling visual variations is crucial.
-   **Advantage over other Transfer/Disentanglement Methods:** DisWM also outperforms `TED` and `APV`. The gap with `TED` suggests that pre-learning semantics from videos is more effective than learning them from scratch online. The advantage over `APV` and `DV2 Finetune` highlights the importance of DisWM's specialized `offline-to-online latent distillation`, which appears to transfer knowledge more effectively than simpler finetuning approaches, especially in the challenging cross-domain $DMC -> MuJoCo$ setting.

## Qualitative Results
The paper provides qualitative visualizations to demonstrate that the model is indeed learning disentangled representations.

-   **Figure 5 (Pretraining):** This figure shows "latent traversals" from the pretrained `β-VAE`. In each row, a single dimension of the latent code is varied while others are kept fixed. The resulting generated images show that a single, interpretable factor of variation (like the color of the cheetah or finger) changes, while other aspects of the scene remain constant. This provides strong visual evidence of successful disentanglement during the pretraining phase.

    ![Figure 5. Visualization of traversals of $\\beta$ VAE during the pretraining phase.](images/5.jpg)
    *该图像是图表，展示了在预训练阶段 `eta` VAE 的遍历情况。上半部分为 Cheetah Color 处理的轨迹，下半部分为 Finger Color 处理的轨迹，显示了不同颜色对应的表现差异。*

-   **Figure 6 (Finetuning):** This figure shows similar latent traversals for the world model during the online finetuning phase on the MuJoCo Pusher task. It demonstrates that the world model successfully disentangles factors like object color, background color, and robot arm color, even while learning a complex control task. This shows the disentanglement capability is maintained and adapted during online learning.

    ![该图像是示意图，展示了不同特征在视频序列中的变化，包括对象、背景颜色和机器人手臂颜色的不同排列。这些变化用于说明在强化学习中如何实现语义知识的转移和学习。](images/6.jpg)
    *该图像是示意图，展示了不同特征在视频序列中的变化，包括对象、背景颜色和机器人手臂颜色的不同排列。这些变化用于说明在强化学习中如何实现语义知识的转移和学习。*

## 6.2. Data Presentation (Tables)
The supplementary material provides additional quantitative comparisons.

The following are the results from Table A of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th>Reacher Easy → Cheetah Run</th>
<th>Walker Walk → Humanoid Walk</th>
</tr>
</thead>
<tbody>
<tr>
<td>DreamerV3</td>
<td>662 ± 9</td>
<td>12 ± 17</td>
</tr>
<tr>
<td>TD-MPC2</td>
<td>510 ± 15</td>
<td>1 ± 0</td>
</tr>
<tr>
<td>ContextWM</td>
<td>661 ± 49</td>
<td>1 ± 0</td>
</tr>
<tr>
<td>DisWM</td>
<td><b>817 ± 59</b></td>
<td><b>147 ± 85</b></td>
</tr>
</tbody>
</table>

This table shows that DisWM outperforms even stronger baselines like `DreamerV3` and `TD-MPC2` on challenging transfer tasks, especially the difficult `Humanoid Walk` task.

The following are the results from Table B of the original paper:

| Model | DrawerClose | DrawerOpen |
| :--- | :--- | :--- |
| TDMPC2 | 3 ± 6 | 43 ± 25 |
| ContextWM | 37 ± 12 | 23 ± 25 |
| DisWM | **77 ± 6** | **70 ± 10** |

This table reports the success rate (%) on the `DrawerWorld` benchmark with texture variations. DisWM achieves a significantly higher success rate, demonstrating its robustness to texture changes, which is a more complex visual variation than simple color changes.

The following are the results from Table C of the original paper:

| Model | Training Steps | Training time | Inference time | Params (M) |
| :--- | :--- | :--- | :--- | :--- |
| CURL | 100k | 303 min | 4.97 sec | 10.7 |
| DV2 FT | 200k | 1522 min | 9.88 sec | 12.1 |
| APV | 200k | 1722 min | 10.15 sec | 13 |
| TED | 100k | 1051 min | 20.49 sec | 11.5 |
| DV2 | 100k | 901 min | 9.59 sec | 12.1 |
| DisWM | 200k | 1311 min | 9.48 sec | **5.8** |

This table provides a runtime and model size comparison. A surprising and highly favorable result for DisWM is that it has **significantly fewer parameters** (5.8M) than all other major baselines (10.7M - 13M). Despite being smaller, it trains in a comparable amount of time and achieves better performance, making it a more efficient model.

## 6.3. Ablation Studies / Parameter Analysis
The paper conducts several experiments to validate the contribution of each component of DisWM.

![该图像是示意图，展示了不同方法在环境步数与回合收益上的对比。左侧图表显示了DisWM模型与不使用蒸馏和不使用解耦的效果；中间图表展示了不同蒸馏权重对表现的影响；右侧图表则表现了不同解耦比例对效果的影响。](images/7.jpg)
*该图像是示意图，展示了不同方法在环境步数与回合收益上的对比。左侧图表显示了DisWM模型与不使用蒸馏和不使用解耦的效果；中间图表展示了不同蒸馏权重对表现的影响；右侧图表则表现了不同解耦比例对效果的影响。*

-   **Ablation Studies (Figure 7, Left):**
    -   **`w/o Distillation` (Green Curve):** Removing the latent distillation loss ($η = 0$) causes a significant drop in performance, especially in the early stages of training. This confirms that the distillation mechanism is crucial for effectively transferring the pretrained knowledge and kick-starting the learning process.
    -   **`w/o Disentanglement` (Blue Curve):** Removing the disentanglement constraints entirely (setting $β$ and $β₂$ to 0) leads to the worst performance. This demonstrates that the core idea of learning and enforcing disentangled representations is fundamental to the method's success in handling visual distractors.

-   **Sensitivity Analyses (Figure 7, Middle & Right):**
    -   **Distillation Weight $η$ (Middle):** The performance is sensitive to the weight of the distillation loss. A very low $η$ fails to transfer enough knowledge, while a very high $η$ can cause the model to overfit to the pretrained teacher and hinder adaptation to the new task's dynamics.
    -   **Disentanglement Scale $β$ (Right):** Similarly, the disentanglement weight $β$ requires careful tuning. A small $β$ is insufficient to enforce disentanglement, while a very large $β$ can harm the model's ability to reconstruct the image accurately, leading to a loss of essential information and degrading performance.
    -   **Latent Space Dimension (Figure I):** The performance is also sensitive to the dimension of the latent space `z_dim`. A very small dimension (e.g., 5) is too restrictive and cannot capture all the factors of variation, leading to poor performance. Increasing the dimension helps, but a very large dimension (e.g., 100) does not necessarily yield further gains and can make training less stable.

-   **Effect of Video Domain (Figure 8):**

    ![Figure 8. Performance of DisWM on DMC Cartpole Swingup with different video datasets.](images/8.jpg)
    *该图像是图表，展示了DisWM在DMC Cartpole Swingup任务中，使用不同视频数据集的表现。横轴为环境步数，纵轴为赛季回报，颜色线条代表不同的数据集，结果表明预训练方法显著提升了性能。*

    This experiment tests how the choice of pretraining video dataset affects performance on the `Cartpole Swingup` task. The results show that pretraining on videos from any of the other tasks (`Finger Spin`, `Reacher Easy`, etc.) provides a significant benefit over `DreamerV2` (which has no pretraining). This indicates that the framework is robust and can extract useful, generalizable semantic knowledge even from out-of-domain videos.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper introduces **Disentangled World Models (DisWM)**, a novel model-based reinforcement learning framework designed to improve the sample efficiency and robustness of VRL agents in environments with visual variations. The core contribution is a pretraining-finetuning paradigm that leverages unlabeled, distracting videos to learn a general understanding of visual semantics. This knowledge is then efficiently transferred to a downstream RL agent via a specialized **offline-to-online latent distillation** technique. By combining this with flexible **disentanglement constraints** during online adaptation, DisWM enables the agent to focus its interactions on learning the control task, rather than re-learning how to interpret a visually complex world. The experimental results strongly validate the approach, showing that DisWM outperforms state-of-the-art methods on various benchmarks, even in challenging cross-domain transfer scenarios, and does so with a more parameter-efficient model.

## 7.2. Limitations & Future Work
The authors acknowledge one primary limitation:
-   **Complexity of Environments:** The current work focuses on relatively controlled variations, such as changes in color, texture, and background. The authors suggest that disentangled representation learning faces greater challenges in more complex and dynamic environments. Future work could explore the framework's effectiveness in non-stationary settings with more intricate variations, such as dealing with background videos or other dynamically moving, irrelevant objects.

## 7.3. Personal Insights & Critique
-   **Strengths:**
    -   The paper's core idea is both intuitive and powerful. Decoupling the learning of "what the world looks like" from "how to act in the world" is a very promising direction for building more general and efficient AI agents.
    -   The combination of `β-VAE` for disentanglement, world models for sample efficiency, and latent distillation for knowledge transfer is a clever and well-executed synthesis of existing powerful techniques.
    -   The finding that DisWM is more parameter-efficient than its competitors (Table C) is a significant practical advantage, making the method more appealing for real-world deployment.
    -   The cross-domain experiments (e.g., DMC to MuJoCo) are particularly compelling, as they demonstrate a non-trivial level of knowledge transfer that goes beyond simple in-domain generalization.

-   **Potential Issues and Areas for Improvement:**
    -   **"Distracting" vs. "In-the-Wild":** While the paper uses the term "distracting videos," the pretraining data is still generated from the same underlying simulators (DMC, MuJoCo), just with programmatic color changes. A true test of this approach would be to pretrain on genuinely "in-the-wild" videos (e.g., from YouTube or robotics datasets like Ego4D) and transfer that knowledge to a simulated or real robot. The domain gap in that scenario would be far greater and would pose a much harder challenge.
    -   **Reliance on β-VAE:** The method's disentanglement capability is tied to the `β-VAE` framework, which is known to have its own limitations. Achieving robust, unsupervised disentanglement is still an open research problem, and `β-VAE` does not always guarantee that the learned factors will align perfectly with human-interpretable semantic concepts.
    -   **Hyperparameter Sensitivity:** The analysis shows that the method's performance is sensitive to key hyperparameters like $β$ and $η$. This might make it challenging to apply the method to a new, unseen problem without careful tuning, which can be computationally expensive.
    -   **Scalability to High-Dimensional Observation:** The experiments are run on `64x64` pixel images. Scaling world models and VAE-based approaches to high-resolution, photorealistic imagery remains a significant challenge due to the difficulty of high-fidelity reconstruction. Exploring this method with more advanced generative models (e.g., diffusion models) could be a fruitful future direction, as hinted at by other recent works.