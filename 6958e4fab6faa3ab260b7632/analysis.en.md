# 1. Bibliographic Information
## 1.1. Title
What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

## 1.2. Authors
- Jimmy Yang (Meta FAIR)
- Basile Terver (Meta FAIR, INRIA Paris)
- Jean Ponce (Ecole normale supérieure/PSL, New York University)
- Adrien Bardes (Meta FAIR)
- Yann LeCun (Meta FAIR)

  The author list includes prominent figures in AI, most notably Yann LeCun, a Turing Award laureate known for his foundational work in deep learning, convolutional neural networks, and the concept of self-supervised learning. The authors are primarily affiliated with Meta's Fundamental AI Research (FAIR) lab, a leading institution in AI research. This indicates the work is well-resourced and positioned at the forefront of current AI developments, particularly in the area of autonomous intelligent agents.

## 1.3. Journal/Conference
The paper is available as a preprint on arXiv. The submission date `2025-12-30T22:50:03.000Z` suggests it is intended for a future top-tier conference in AI or robotics, such as ICLR, NeurIPS, ICML, or CoRL, which are highly competitive and influential venues.

## 1.4. Publication Year
The publication date on arXiv is listed as 2025, indicating it is a very recent work submitted for review for a 2026 conference cycle. The version analyzed is $v1$.

## 1.5. Abstract
The paper investigates the factors contributing to the success of a class of world models used for physical planning, which the authors term `JEPA-WMs` (Joint-Embedding Predictive Architecture World Models). These models learn from state-action data and perform planning in a learned representation (latent) space, rather than the raw input (e.g., pixel) space. The goal is to understand which design choices—in model architecture, training objective, and planning algorithm—are most effective. Through a comprehensive study involving simulated environments and real-world robotic data, the authors analyze several key components. They consolidate their findings to propose an optimized `JEPA-WM` that surpasses established baselines like `DINO-WM` and `V-JEPA-2-AC` on both navigation and manipulation tasks.

## 1.6. Original Source Link
- **Original Source Link:** `https://arxiv.org/abs/2512.24497`
- **PDF Link:** `https://arxiv.org/pdf/2512.24497v1.pdf`
- **Publication Status:** This is a preprint on arXiv and has not yet been peer-reviewed or officially published in a conference or journal.

# 2. Executive Summary
## 2.1. Background & Motivation
The core problem is to create autonomous agents that can effectively solve a wide variety of physical tasks, such as robotic navigation and manipulation, and generalize to new situations. A promising approach is to use **world models**, which are internal models of how the world evolves in response to actions. An agent can use this model to "imagine" the future and plan a sequence of actions to achieve a goal.

While many world models predict future states in the raw input space (e.g., generating future video frames), this can be computationally expensive and waste capacity on irrelevant details (e.g., lighting changes, background textures). A newer family of methods, which this paper calls **`JEPA-WMs`**, performs prediction and planning in an abstract **latent space**. The intuition is that a good latent space captures only the essential information needed for the task, making planning more efficient and robust.

Despite the growing popularity of `JEPA-WMs`, there was no systematic study that identifies which specific design choices are crucial for their success. Different papers use different architectures, training losses, and planning algorithms, making it difficult to understand the key drivers of performance. This paper aims to fill that gap by conducting a rigorous and comprehensive ablation study to find the "optimal" configuration for `JEPA-WMs` in physical planning tasks.

## 2.2. Main Contributions / Findings
The paper's main contributions are:

1.  **Systematic Investigation of `JEPA-WMs`:** The authors conduct a thorough empirical study on the key components of `JEPA-WMs`, isolating the impact of:
    *   **Planning Algorithm:** Comparing sampling-based methods (`CEM`, `Nevergrad`) and gradient-based methods (`Adam`, `GD`).
    *   **Training Objective:** Analyzing the effect of `multistep rollout` losses during training.
    *   **Model Architecture:** Investigating predictor architecture, action conditioning methods (`feature conditioning`, `sequence conditioning`, `AdaLN`), and model scaling (encoder/predictor size and depth).
    *   **Input Modalities:** Studying the importance of including `proprioception` (e.g., robot joint angles) alongside visual input.
    *   **Pretrained Encoders:** Comparing different frozen visual encoders (`DINOv2`, `DINOv3`, `V-JEPA`, `V-JEPA-2`).

2.  **Proposal of an Optimized `JEPA-WM`:** Based on their findings, the authors combine the best-performing components to create a new model that outperforms two strong baselines, `DINO-WM` and `V-JEPA-2-AC`, on a range of navigation and manipulation benchmarks.

3.  **Key Findings:**
    *   **Planners:** The sampling-based `Cross-Entropy Method (CEM)` with an $L_2$ distance metric is the best all-around planner. Gradient-based planners excel in simple environments but fail in complex ones with local minima. `Nevergrad` is a good alternative to `CEM` that requires less hyperparameter tuning.
    *   **Training:** A `2-step rollout` loss improves performance on simulated data, while real-world data benefits from even longer rollouts (e.g., 6-step).
    *   **Encoders & Input:** Pretrained image encoders with strong object segmentation features (`DINOv2`, `DINOv3`) outperform video encoders (`V-JEPA`). Including `proprioception` consistently improves performance.
    *   **Scaling:** Larger models (bigger encoders and deeper predictors) provide significant benefits on complex, real-world data (`DROID`) but offer no advantage (and can even be detrimental) on simpler, simulated tasks.
    *   **Architecture:** `AdaLN` conditioning is a robust and efficient choice for injecting action information into the predictor.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
### 3.1.1. World Models
A **world model**, a concept popularized by Ha & Schmidhuber (2018), is a model learned by an agent to simulate its environment. Given the current state of the world and a proposed action, the world model predicts the next state. This allows the agent to perform "mental simulation" or "imagination." Instead of having to try out actions in the real world (which can be slow, expensive, or dangerous), the agent can use its internal world model to plan ahead, anticipate outcomes, and choose the best sequence of actions to achieve a goal. World models are a cornerstone of **model-based reinforcement learning (MBRL)**.

### 3.1.2. Joint-Embedding Predictive Architectures (JEPA)
Proposed by Yann LeCun (2022), **JEPA** is a self-supervised learning paradigm for learning representations. Unlike generative models that try to reconstruct every detail of an input (e.g., predicting all pixels in an image), JEPAs operate in an abstract representation space. The core idea is to predict the representation of one part of the input from the representation of another part.

The architecture consists of:
*   An **encoder** that maps the input data (e.g., an image) to a representation (embedding).
*   A **predictor** that takes the representation of a *context* block and predicts the representation of a *target* block in the same input.
*   A **loss function** that minimizes the distance between the *predicted* representation and the *actual* representation of the target block (computed by the same encoder).

    By avoiding pixel-level reconstruction, JEPAs are encouraged to learn more abstract and semantic features, ignoring irrelevant low-level details. This paper extends this idea to dynamic environments, where the "context" is the past states and actions, and the "target" is the future state.

### 3.1.3. Model Predictive Control (MPC)
**Model Predictive Control (MPC)** is a planning strategy where an agent repeatedly plans over a short, finite time horizon. The process is as follows:
1.  At the current state, the agent uses its world model to find the optimal sequence of actions over a fixed horizon (e.g., the next 5 seconds).
2.  The agent executes only the *first* action (or first few actions) from this sequence.
3.  The agent observes the new state of the world.
4.  The agent discards the rest of the old plan and repeats the process from step 1, re-planning from the new state.

    This "plan, execute, observe, replan" loop makes the agent robust to prediction errors from its world model and unexpected changes in the environment. It is the core planning paradigm used in this paper.

### 3.1.4. Vision Transformer (ViT)
A **Vision Transformer (ViT)**, introduced by Dosovitskiy et al. (2021), is an architecture that applies the Transformer model, originally designed for natural language processing, to computer vision tasks. It works by:
1.  Splitting an image into a sequence of fixed-size, non-overlapping patches.
2.  Linearly embedding each patch into a vector.
3.  Adding positional information to these patch embeddings.
4.  Feeding the resulting sequence of vectors into a standard Transformer encoder.

    The Transformer's `self-attention` mechanism allows the model to weigh the importance of different patches when processing any given patch, enabling it to learn global relationships within the image. The models in this paper use ViTs as both the visual encoder and the dynamics predictor.

## 3.2. Previous Works
The paper situates itself within the context of recent advancements in world modeling and robotics.
*   **Generative World Models:** Works like `Genie` (Bruce et al., 2024) and `GAIA-1` (Hu et al., 2023) focus on generating high-fidelity future video frames. While visually impressive, they are computationally intensive and may not be the most efficient for planning.
*   **Latent-Space World Models (Non-JEPA):** This is a large family of models from the reinforcement learning community.
    *   `Dreamer` series (Hafner et al., 2024): Learns a compact latent dynamics model and uses it to train a policy entirely within the "dreamed" environment. It typically relies on reward signals for training.
    *   `TD-MPC` (Hansen et al., 2024): Combines a learned world model with a value function to perform planning that maximizes future rewards.
*   **Latent-Space World Models (JEPA family):** These are the direct predecessors and baselines for this work.
    *   `PLDM` (Sobal et al., 2025): Showed that a JEPA-style world model trained on suboptimal data can generalize better than many goal-conditioned RL methods.
    *   `DINO-WM` (Zhou et al., 2024a): A key baseline. It uses a frozen, pretrained `DINOv2` encoder to provide visual features and learns a predictor on top of these features. It demonstrated strong zero-shot planning capabilities without requiring reward signals.
    *   `V-JEPA-2-AC` (Assran et al., 2025): Another key baseline. It also uses a JEPA approach for object manipulation tasks, showing competitiveness against large Vision-Language-Action (VLA) models.

## 3.3. Technological Evolution
The field has evolved from model-free RL (which requires massive interaction data) towards model-based RL to improve sample efficiency. Early model-based approaches often struggled with compounding errors in long-term predictions. The rise of self-supervised learning and large pretrained models (like `DINOv2`) provided a breakthrough. Instead of learning a visual representation from scratch, recent methods leverage these powerful encoders, which already understand objects, geometry, and semantics. The focus has thus shifted from learning the *representation* to learning the *dynamics* in that representation space. The `JEPA` paradigm offers a principled way to do this without the overhead of generative modeling, making it an attractive and rapidly developing area.

## 3.4. Differentiation Analysis
Compared to related work, this paper's core innovation is not a single new algorithm but rather a **methodical and comprehensive scientific investigation**.
*   **Not a Novel Algorithm, but a "Best Practices" Guide:** Unlike papers that propose a brand-new architecture, this work takes an existing family of models (`JEPA-WMs`) and systematically dissects it to understand what makes it work. Its contribution is knowledge and guidance for the research community.
*   **Broad Scope of Analysis:** It simultaneously investigates multiple facets of the problem—planning, training, architecture, and data—which are often studied in isolation. This holistic view provides a more complete picture of the design trade-offs.
*   **Focus on Reward-Free Planning:** Like its direct predecessors (`DINO-WM`), the paper focuses on the reward-free, goal-conditioned setting. An agent is given a starting image and a goal image and must plan to reach the goal, without any intermediate reward signals. This is a more general and scalable setup than methods relying on hand-crafted reward functions.

# 4. Methodology
The paper formalizes the family of **Action-conditioned Joint-Embedding Predictive World Models (`JEPA-WMs`)** and investigates its components. The methodology can be broken down into two main phases: training the world model and using it for planning.

![Figure 1: Left: Training of JEPA-WM: the encoder $E _ { \\phi , \\theta }$ embeds video and optionally proprioceptive observation, which is fed to the predictor $P _ { \\theta }$ , along with actions, to predict (in parallel across timesteps) the next state embedding. Right: Planning with JEPA-WM: sample action sequences, unroll the predictor on them, compute a planning cost $L ^ { p }$ for each trajectory, and use this cost to $A _ { \\theta }$ and proprioceptive encoder $E _ { \\theta } ^ { p r o p }$ are not explicitly displayed in this figure for readability.](images/1.jpg)
*该图像是示意图，展示了JEPA-WM模型的训练和规划过程。左侧部分展示了通过编码器$E_{\theta}$对视频和动作进行嵌入，预测器$P_{\theta}$并行预测下一个状态嵌入，并使用JEPA教师强制损失$L$进行训练。右侧展示了通过样本动作序列展开预测器，计算每条轨迹的规划成本$L^p$，以优化规划。*
*Figure 1: This figure provides a high-level overview of the two phases. **Left (Training):** The model learns to predict future representations. The encoder $E$ embeds observations (video and proprioception), and the predictor $P$ takes these embeddings and actions to predict the next state's embedding. The loss compares this prediction to the ground-truth embedding of the next state. **Right (Planning):** To find a plan, the model samples many different action sequences, uses the learned predictor to "unroll" or simulate the future for each sequence, and calculates a cost (distance to the goal embedding) for each resulting trajectory. The best action sequence is then selected.*

## 4.1. Principles
The core principle is to learn a forward dynamics model in a latent space provided by a powerful, pretrained, and **frozen** visual encoder. The model does not predict future pixels but rather future latent representations. The training objective is to make the predicted latent state at time $t+1$ as close as possible to the true latent state at time $t+1$. Planning is then framed as an optimization problem: find the sequence of actions that minimizes the distance between the predicted latent state at the end of the plan and the latent representation of a given goal image.

## 4.2. Core Methodology In-depth (Layer by Layer)
### 4.2.1. Model Architecture
The `JEPA-WM` consists of several key components:
*   **Observation Encoder ($E_{\phi, \theta}$):** This component maps raw observations to latent representations. It is composed of:
    *   A **Visual Encoder ($E_\phi^{vis}$):** A frozen, pretrained Vision Transformer (e.g., `DINOv2`). It takes an image observation $o_t^{vis}$ and outputs a set of patch tokens (a latent representation). The weights $\phi$ are not updated during training.
    *   A **Proprioceptive Encoder ($E_\theta^{prop}$):** An optional, shallow network (e.g., a linear layer) that embeds the robot's physical state $o_t^{prop}$ (e.g., joint angles, gripper position). Its weights $\theta$ are trained.
*   **Action Encoder ($A_\theta$):** A network (e.g., a linear layer) that embeds the action vector $a_t$ into a latent representation. Its weights $\theta$ are trained.
*   **Predictor ($P_\theta$):** This is the core dynamics model, typically a Transformer architecture. It takes a context of past state representations and action representations and predicts the state representation for the next timestep. Its weights $\theta$ are trained.

### 4.2.2. Training the World Model
The world model is trained on a dataset of state-action trajectories. The training objective is to minimize the prediction error in the latent space.

For a batch of $B$ trajectory segments, the fundamental training loss is given by Equation (1) in the paper:

\$
\mathcal { L } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } L [ P _ { \theta } \left( E _ { \phi , \theta } ( o _ { t - w : t } ^ { b } ) , A _ { \theta } ( a _ { t - w : t } ^ { b } ) \right) , E _ { \phi , \theta } \left( o _ { t + 1 } ^ { b } \right) ]
\$

Let's break down this formula step-by-step:
1.  **Input Context:** The model takes a window of past observations $o_{t-w:t}^b = (o_{t-w}^b, \dots, o_t^b)$ and past actions $a_{t-w:t}^b = (a_{t-w}^b, \dots, a_t^b)$. Here, $w$ is the context window length.
2.  **Encoding:**
    *   The observation encoder $E_{\phi, \theta}$ processes the observation sequence $o_{t-w:t}^b$ to get a sequence of latent state representations.
    *   The action encoder $A_\theta$ processes the action sequence $a_{t-w:t}^b$ to get a sequence of latent action representations.
3.  **Prediction:** The predictor $P_\theta$ takes these encoded past states and actions as input and outputs a prediction for the latent state at the next timestep, $t+1$.
4.  **Target:** The observation encoder $E_{\phi, \theta}$ is also used to compute the "ground truth" latent representation for the next observation, $E_{\phi, \theta}(o_{t+1}^b)$. This is known as **teacher-forcing**, as the model is always given the true previous states to make its next prediction.
5.  **Loss Calculation:** The loss function $L$ (typically Mean Squared Error, MSE) computes the dissimilarity between the predicted latent state and the target latent state. If both visual and proprioceptive modalities are used, the loss is computed separately for each and then combined.
6.  **Averaging:** The final loss $\mathcal{L}$ is the average over all $B$ samples in the batch.

    The predictor $P_\theta$ is trained with a causal attention mask, which allows it to be trained simultaneously to predict from all context lengths from 0 to `W-1` (where $W$ is the max training context size).

### 4.2.3. Planning with the World Model
Once the world model is trained, it is used for planning via Model Predictive Control (MPC). Given a current observation $o_t$ and a goal observation $o_g$, planning is an optimization problem over a sequence of future actions $a_{t:t+H-1} = (a_t, \dots, a_{t+H-1})$ of length $H$.

The objective is to minimize a planning cost, defined in Equation (2):

\$
L _ { \alpha } ^ { p } ( o _ { t } , a _ { t : t + H - 1 } , o _ { g } ) = ( L _ { v i s } + \alpha L _ { p r o p } ) ( G _ { \phi , \theta } ( o _ { t } , a _ { t : t + H - 1 } ) , E _ { \phi , \theta } ( o _ { g } ) )
\$

Here's the breakdown:
1.  **Goal Encoding:** First, the goal observation $o_g$ is encoded into a target latent representation $E_{\phi, \theta}(o_g)$.
2.  **Imagined Trajectory Generation:** For a candidate action sequence $a_{t:t+H-1}$, the function $G_{\phi, \theta}$ simulates the future. This function represents the "unrolling" of the predictor. It starts with the latent representation of the current state, $z_t = E_{\phi, \theta}(o_t)$, and recursively applies the predictor $P_\theta$ for $H$ steps to generate a sequence of predicted future latent states, culminating in the final predicted state $\hat{z}_{t+H}$. This unrolling process is detailed in Equations (3) and (4).
3.  **Recursive Unrolling:** The unrolling function $F_{\phi, \theta}$ (which is what $G_{\phi, \theta}$ is in this paper) is defined recursively. To get the prediction for time $i+1$, the predictor uses a sliding window of the most recent $w$ predicted states and actions.
    \$
    \hat{z}_{i+1} = P_{\theta}(\hat{z}_{i-w:i}, A_{\theta}(a_{i-w:i})), \quad \text{for } i = t, \dots, t+k-1
    \$
    Here, $\hat{z}_{i-w:i}$ is a sequence of previously predicted embeddings (or ground-truth embeddings if they are part of the initial context). The process starts with $\hat{z}_t = E_{\phi, \theta}(o_t)$.
4.  **Cost Calculation:** The dissimilarity metric $L$ (e.g., $L_1$ or $L_2$ distance) is applied between the final predicted latent state from the unrolling, $G_{\phi, \theta}(o_t, a_{t:t+H-1})$, and the target goal representation, $E_{\phi, \theta}(o_g)$. The total cost $L_\alpha^p$ is a weighted sum of the visual distance ($L_{vis}$) and the proprioceptive distance ($L_{prop}$), with $\alpha$ controlling the weight of the proprioceptive component.
5.  **Optimization:** A planning algorithm (like CEM or Adam) is used to search for the action sequence $a_{t:t+H-1}$ that minimizes this cost $L_\alpha^p$.

### 4.2.4. Multistep Rollout Training
The standard "teacher-forcing" training (Equation 1) can lead to a mismatch between training and testing. During training, the predictor always sees ground-truth embeddings, but during planning, it must make predictions based on its own previous (and possibly imperfect) predictions. This can cause errors to accumulate rapidly.

To mitigate this, the authors explore adding a **multistep rollout loss**. This involves unrolling the predictor for $k$ steps during training, using its own outputs as inputs, and then computing a loss at the end. The $k$-step rollout loss $\mathcal{L}_k$ is defined in Equation (5):

\$
\mathcal { L } _ { k } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } L [ P _ { \theta } ( \hat { z } _ { t - w : t + k - 1 } ^ { b } , A _ { \theta } ( a _ { t - w : t + k - 1 } ^ { b } ) ) , E _ { \phi , \theta } \left( o _ { t + k } ^ { b } \right) ]
\$

*   Here, $\hat{z}_{t-w:t+k-1}^b$ is a sequence containing ground-truth embeddings up to time $t$ and predicted embeddings from time $t+1$ to `t+k-1`.
*   The model predicts the state at time $t+k$ and compares it to the ground-truth embedding $E_{\phi, \theta}(o_{t+k}^b)$.
*   The total training loss can be a sum of losses for different rollout lengths, e.g., $\mathcal{L}_{total} = \mathcal{L}_1 + \mathcal{L}_2 + \dots$. This forces the model to learn to make stable predictions over longer horizons, better aligning the training and planning procedures.

# 5. Experimental Setup
## 5.1. Datasets
The study uses a diverse set of datasets covering both simulated and real-world robotics tasks.
*   **Metaworld:** A benchmark for multi-task and meta-reinforcement learning featuring a Sawyer robotic arm performing 50 different manipulation tasks. The authors collected their own dataset using a `TD-MPC2` agent and evaluate on "Reach" and "Reach-Wall" tasks.
*   **Push-T:** A simulated task where a pusher ball must push a T-shaped block to a target location. This requires precise interaction.
*   **PointMaze & Wall:** Simple 2D navigation tasks. `PointMaze` involves a ball navigating a maze, and `Wall` involves an agent navigating between two rooms through a door. These test basic planning and collision avoidance.
*   **DROID (Large-scale In-the-wild Robot manipulation Dataset):** A large-scale, real-world dataset of a Franka Panda arm performing various manipulation tasks in diverse settings, collected via teleoperation. This is a key dataset for evaluating real-world performance.
*   **Robocasa:** A simulated environment designed to mimic everyday household tasks. The authors use it for zero-shot evaluation of models trained on `DROID`, testing how well the learned skills transfer from the real world to a new simulated one.

## 5.2. Evaluation Metrics
The primary metric is **Success Rate**, which measures the percentage of episodes where the agent successfully completes the task (e.g., reaching the goal position within a certain tolerance).

However, success rate can be noisy and sparse. Therefore, the authors also track several other metrics to evaluate the quality of the world model itself:
*   **Embedding Space Error:** The $L_1$ or $L_2$ distance between the predicted latent embeddings and the ground-truth future embeddings during an open-loop rollout. This directly measures the accuracy of the world model.
*   **Proprioceptive Decoding Error:** The error between the predicted proprioceptive state (decoded from the latent space) and the ground-truth future state.
*   **Visual Decoding Quality (LPIPS):** A visual decoder is trained to reconstruct images from latent embeddings. The authors use this to visualize the "imagined" futures and quantitatively measure their perceptual similarity to the ground-truth future frames using the LPIPS metric.
    *   **Conceptual Definition:** The **Learned Perceptual Image Patch Similarity (LPIPS)** metric evaluates the perceptual similarity between two images. Unlike pixel-wise metrics like MSE, which are sensitive to small shifts and texture variations, LPIPS is designed to align better with human perception of similarity. It works by feeding two images through a deep neural network (e.g., VGG or AlexNet) and computing the distance between their internal feature activations at different layers.
    *   **Mathematical Formula:**
        \$
        d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} || w_l \odot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l) ||_2^2
        \$
    *   **Symbol Explanation:**
        *   $x, x_0$: The two images being compared.
        *   $l$: Index of a specific layer in the deep network.
        *   $\hat{y}^l, \hat{y}_0^l$: Activations from layer $l$ for images $x$ and $x_0$, respectively, normalized across the channel dimension.
        *   $H_l, W_l$: Height and width of the feature map at layer $l$.
        *   $w_l$: A scaling vector used to weight the importance of each channel at layer $l$.
*   **Action Score (for DROID):** Since success is hard to define on the static DROID dataset, the authors measure the $L_1$ error between the actions planned by the model and the ground-truth human-teleoperated actions. This error is then rescaled to create a score to be maximized.

## 5.3. Baselines
The paper's proposed optimal model is compared against two state-of-the-art `JEPA-WM` baselines:
*   **`DINO-WM` (Zhou et al., 2024a):** A model that uses a frozen `DINOv2` encoder and learns a predictor via feature conditioning. It represents a strong baseline for zero-shot planning. The authors use `DINO-WM` as their base configuration for the ablation study.
*   **`V-JEPA-2-AC` (Assran et al., 2025):** A powerful `JEPA-WM` that uses sequence conditioning and a video-centric `V-JEPA-2` encoder, achieving strong results in object manipulation.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The authors systematically test each design choice. Below is a summary of the key findings, often referencing the figures in the paper.

### 6.1.1. Planner Comparison (Figure 3, Left)
The choice of optimization algorithm for planning is critical.
*   The sampling-based **`Cross-Entropy Method (CEM)` with an $L_2$ cost function** emerges as the most robust and best-performing planner across the board.
*   **Gradient-based planners (`Adam`, `GD`)** perform well on simple tasks with smooth cost landscapes (like Metaworld) but fail catastrophically on tasks requiring non-greedy navigation (Wall, Maze) or complex contact dynamics (DROID), as they get stuck in local minima.
*   The **`Nevergrad (NG)`** planner, another sampling-based method, performs on par with `CEM` on the complex real-world datasets (`DROID`, `Robocasa`) and has the significant advantage of requiring no hyperparameter tuning, making it a practical choice.
*   Across all planners, using the **$L_2$ distance** in the planning cost function consistently outperforms the $L_1$ distance.

    ![Figure 3: Left: Comparison of planning optimizers: NG is the Nevergrad-based interface for trajectory optimization that we introduce, compared to the Cross-Entropy Method (CEM), with `L _ { 1 }` or `L _ { 2 }` distance. Right: Effect of adding multistep rollout loss terms: models are trained with total loss $\\mathcal { L } _ { 1 } + \\cdots + \\mathcal { L } _ { k }$ . Rc-Pl and Rc-R denote the Place and Reach tasks of Robocasa.](images/3.jpg)
    *该图像是图表，展示了不同规划优化器的性能比较和多步回滚损失项的影响。左侧柱状图比较了CEM、NG及其他模型在各种任务（如MW-Reach、Maze、Push-T等）中的表现，性能以百分比表示。右侧折线图显示了在不同回滚步骤下的平均性能。整个图表突出了不同优化算法在物理任务中的效果和稳定性。*
    *Figure 3: **Left:** This bar chart compares different planning optimizers across various tasks. The `CEM L2` (blue bar) consistently achieves high success rates across most environments, establishing it as the strongest overall choice. Gradient-based methods (e.g., `Adam L2`, orange bar) excel on `MW-R` but perform poorly on navigation tasks like `Wall` and `Push-T`. **Right:** This line chart shows that adding a `multistep rollout` loss improves performance. Performance generally peaks at a `2-step` loss for simulated environments, while it continues to improve for more steps on the real-world DROID dataset (not shown in this chart but mentioned in the text).*

### 6.1.2. Impact of Proprioception and Encoder Type (Figure 4)
*   **Proprioception:** Adding proprioceptive input (`prop`) consistently improves performance over using visual input alone (`no-prop`). This is especially true for tasks where precise control is needed, as it helps the agent avoid oscillating around the goal.
*   **Encoder Type:** Pretrained **image encoders (`DINOv2`, `DINOv3`)** significantly outperform pretrained **video encoders (`V-JEPA`, `V-JEPA-2`)**. The authors hypothesize this is because the DINO family has superior object segmentation and fine-grained spatial understanding, which is crucial for manipulation and navigation. `DINOv3` shows a particular advantage on photorealistic data (`Robocasa`, `DROID`).

    ![Figure 4: Left: Models trained with proprioceptive input are denoted "prop", while pure visual world models are named "no-prop". Right: Comparison of JEPA-WMs trained on top of various pretrained visual encoders, all of size ViT-L for fair comparison. Rc-Pl and Rc-R denote the Place and Reach tasks of Robocasa.](images/4.jpg)
    *该图像是图表，展示了用不同输入和预训练视觉编码器训练的JEPA-WM模型在多个任务上的表现对比。左侧显示了具有本体感知输入（prop）和无本体感知输入（no-prop）的模型在 MW-Reach、Maze 等任务上的性能；右侧展示了使用多种预训练编码器的JEPA-WM模型在相同任务上的比较。*
    *Figure 4: **Left:** This chart clearly shows that models trained with proprioceptive input (`prop` bars) outperform their visual-only counterparts (`no-prop` bars) across all simulated tasks. **Right:** This chart compares different frozen encoders. The `DINOv2` and `DINOv3` based models achieve the highest success rates, substantially outperforming the `V-JEPA` and `V-JEPA-2` based models, highlighting the importance of the encoder's feature quality.*

### 6.1.3. Model Scaling and Architecture (Figures 5 & 6)
*   **Model Size:** On simulated environments, scaling up the model (from `ViT-S` to `ViT-L`) does **not** improve performance and can even be harmful. However, on the complex real-world `DROID` dataset, a clear positive correlation is observed: larger encoders and deeper predictors lead to better performance. This suggests that model capacity should be matched to task complexity.
*   **Predictor Depth:** The optimal predictor depth for most simulated tasks is around 6 layers. Deeper is not always better. For `DROID`, a depth of 12 performs best.
*   **Action Conditioning:** The method for injecting action information matters. `AdaLN` (Adaptive Layer Normalization) conditioning performs best on average, likely because it allows action information to modulate every layer of the predictor. However, the best method is task-dependent.

    ![Figure 5: Left: Maximum number of timesteps of state embedding seen by the predictor at train time in equation 1, the predictor takes up to $( E _ { \\phi , \\theta } ( o _ { t - W + 1 : t } ) , A _ { \\theta } ( \\bar { a } _ { t - W + 1 : t } ) )$ as context. Right: Comparison of model size: we vary from ViT-S to ViT-L the visual encoder size, as well as the predictor embedding dimension, keeping predictor depth constant at 6. Rc-Pl and $\\operatorname { R c - R }$ denote the Place and Reach tasks of Robocasa.](images/5.jpg)
    *该图像是图表，显示了不同时间步长 $W$ 和模型大小对任务性能的影响。在左侧图中，性能随时间步长的变化而波动，在右侧图中，模型大小的变化对性能影响较小。多种任务如 MW-Reach-Wall、MW-Reach 和 Push-T 的性能被比较。*
    *Figure 5: **Left (Context Size):** Performance increases significantly when moving from a context size of $W=1$ to $W=2$ or $W=3$, indicating that inferring velocity is important. However, using a very long context can hurt performance on simulated data. **Right (Model Size):** This chart shows that for simulated tasks, increasing the model size from Small ($S$) to Base ($B$) to Large ($L$) does not yield improvements. The performance saturates or slightly degrades.*

### 6.1.4. Proposed Optimal Model
By combining these insights, the authors propose an optimized `JEPA-WM`.
*   **For simulated tasks:** `DINOv2 ViT-S` encoder, predictor of depth 6 with `AdaLN` conditioning, trained with proprioception and a `2-step` rollout loss.
*   **For real-world tasks (DROID/Robocasa):** `DINOv3 ViT-L` encoder, predictor of depth 12, trained with a `6-step` rollout loss (without proprioception for zero-shot transfer compatibility).

    The proposed model is evaluated against the baselines.

## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th>Maze</th>
<th>Wall</th>
<th>Push-T</th>
<th>MW-R</th>
<th>MW-RW</th>
<th>Rc-R</th>
<th>Rc-Pl</th>
<th>DROID</th>
</tr>
</thead>
<tbody>
<tr>
<td>DWM</td>
<td>81.6 (3.4)</td>
<td>64.1 (4.6)</td>
<td>66.0 (4.7)</td>
<td>44.8 (8.9)</td>
<td>35.1 (9.4)</td>
<td>19.1 (13.4)</td>
<td>21.7 (7.2)</td>
<td>39.4 (2.1)</td>
</tr>
<tr>
<td>VJ2AC</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>16.2 (8.3)</td>
<td>33.1 (7.2)</td>
<td>42.9 (2.5)</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>83.9 (2.3)</b></td>
<td><b>78.8 (3.9)</b></td>
<td><b>70.2 (2.8)</b></td>
<td><b>58.2 (9.3)</b></td>
<td><b>41.6 (10.0)</b></td>
<td><b>25.4 (16.6)</b></td>
<td>30.7 (8.0)</td>
<td><b>48.2 (1.8)</b></td>
</tr>
</tbody>
</table>

**Analysis of Table 1:** This table summarizes the final comparison of the authors' optimized model ("Ours") against the baselines `DINO-WM` ("DWM") and `V-JEPA-2-AC` ("VJ2AC"). The numbers represent success rates (or Action Score for DROID), with standard deviations in parentheses. The "Ours" model achieves the best performance on nearly all tasks, often by a significant margin. For example, it improves success on the `Wall` task from 64.1% to 78.8% and on the `Metaworld-Reach` (MW-R) task from 44.8% to 58.2% compared to `DINO-WM`. It also outperforms both baselines on the real-world `DROID` benchmark. This validates the effectiveness of the design choices identified in the ablation studies.

## 6.3. Ablation Studies / Parameter Analysis
The entire paper is structured as a large-scale ablation study. One detailed example is the comparison across all planners and models, provided in Table S5.1 in the appendix.

The following are the results from Table S5.1 of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th>Planner</th>
<th>Maze</th>
<th>Wall</th>
<th>Push-T</th>
<th>MW-R</th>
<th>MW-RW</th>
<th>Rc-R</th>
<th>Rc-Pl</th>
<th>DROID</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="12"><b>DWM</b></td>
<td>CEM L2</td>
<td>81.6 (3.4)</td>
<td>64.1 (4.6)</td>
<td>66.0 (4.7)</td>
<td>44.8 (8.9)</td>
<td>35.1 (9.4)</td>
<td>19.1 (13.4)</td>
<td>21.7 (7.2)</td>
<td>39.4 (2.1)</td>
</tr>
<tr>
<td>CEM L1</td>
<td>78.8 (2.9)</td>
<td>48.7 (4.0)</td>
<td>61.6 (4.5)</td>
<td>45.1 (10.4)</td>
<td>34.0 (9.1)</td>
<td>14.4 (11.3)</td>
<td>19.6 (7.4)</td>
<td>41.7 (2.7)</td>
</tr>
<tr>
<td>NG L2</td>
<td>54.2 (3.8)</td>
<td>25.3 (4.2)</td>
<td>47.6 (5.4)</td>
<td>28.1 (7.8)</td>
<td>27.9 (10.0)</td>
<td>25.8 (16.7)</td>
<td>27.1 (8.6)</td>
<td>36.0 (3.6)</td>
</tr>
<tr>
<td>NG L1</td>
<td>52.3 (4.0)</td>
<td>24.6 (5.2)</td>
<td>46.2 (5.1)</td>
<td>27.5 (8.5)</td>
<td>28.6 (8.9)</td>
<td>21.6 (15.5)</td>
<td>26.0 (8.0)</td>
<td>35.4 (3.3)</td>
</tr>
<tr>
<td>Adam L2</td>
<td>14.8 (1.5)</td>
<td>0.1 (0.3)</td>
<td>8.0 (2.4)</td>
<td>62.0 (8.3)</td>
<td>49.9 (9.9)</td>
<td>2.8 (3.7)</td>
<td>21.5 (8.6)</td>
<td>0.0 (0.0)</td>
</tr>
<tr>
<td>Adam L1</td>
<td>12.6 (2.8)</td>
<td>0.1 (0.3)</td>
<td>2.1 (1.5)</td>
<td>50.7 (8.5)</td>
<td>37.2 (8.3)</td>
<td>0.9 (1.4)</td>
<td>22.1 (7.5)</td>
<td>0.0 (0.0)</td>
</tr>
<tr>
<td>GD L2</td>
<td>14.5 (1.9)</td>
<td>0.1 (0.3)</td>
<td>8.1 (2.4)</td>
<td>62.2 (8.2)</td>
<td>50.0 (10.1)</td>
<td>2.9 (3.8)</td>
<td>21.9 (8.4)</td>
<td>0.0 (0.0)</td>
</tr>
<tr>
<td>GD L1</td>
<td>12.4 (2.6)</td>
<td>0.1 (0.3)</td>
<td>2.0 (1.3)</td>
<td>50.9 (8.5)</td>
<td>37.1 (8.4)</td>
<td>1.0 (1.4)</td>
<td>22.2 (7.3)</td>
<td>0.0 (0.0)</td>
</tr>
<tr>
<td colspan="9">... (VJ2AC and Ours models follow similar structure) ...</td>
</tr>
</tbody>
</table>

*Note: Due to the extreme length of the full table, only the `DWM` section is transcribed here to demonstrate the structure. The full table in the paper details results for `VJ2AC` and `Ours` as well.*

**Analysis of Table S5.1:** This detailed table reinforces the conclusions from the main paper. It shows how performance varies drastically with the choice of planner. For example, for the `DWM` model on the `Wall` task, `CEM L2` achieves 64.1% success, while `Adam L2` and `GD L2` achieve only 0.1%. This highlights the complete failure of gradient-based methods on this navigation task. The table also confirms that the "Ours" model (not fully shown here) consistently outperforms the baselines across most planner configurations, demonstrating its robustness.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
The paper provides a comprehensive and valuable investigation into the design choices that determine the success of `JEPA-WMs` for physical planning. The authors systematically evaluated components related to the planner, training objective, model architecture, and input data. Their key findings indicate that a combination of a strong, pretrained image encoder ($DINOv2/v3$), the inclusion of proprioception, a modest multistep rollout loss, an appropriately scaled model for the task's complexity, and a robust sampling-based planner (`CEM L2`) is critical for high performance. By synthesizing these findings, they constructed an optimized `JEPA-WM` that sets a new state-of-the-art on several robotic planning benchmarks, outperforming established methods.

## 7.2. Limitations & Future Work
The authors provide a thorough analysis, but some limitations and future directions can be inferred:
*   **Planner Limitations:** While `CEM` works well, it is still a simple sampling-based method. More advanced planning algorithms could potentially yield better performance or sample efficiency. The paper also notes that planning success is not guaranteed even with an accurate world model, as the optimization can still fail.
*   **Long-Horizon Tasks:** The planning horizon $H$ in the experiments is relatively short. Solving complex, long-horizon tasks that require hierarchical reasoning remains a major challenge. The current MPC approach may be insufficient for tasks that require a fundamentally different strategy that cannot be found with local optimization.
*   **Data Dependency:** The performance, especially of larger models, is highly dependent on the complexity and diversity of the training data. The gap between simulated and real-world performance suggests that sim-to-real transfer is still an open problem.
*   **Future Work:** The authors release their code and benchmarks, paving the way for future research to build upon their findings. Exploring more sophisticated planners, investigating methods for hierarchical planning, and developing techniques to better bridge the sim-to-real gap are all promising avenues. The introduction of a `Nevergrad` interface for planning also opens up exploration into a wider variety of black-box optimizers.

## 7.3. Personal Insights & Critique
This paper is an excellent example of "good science" in a field often dominated by the pursuit of novel architectures. Instead of just presenting a new model, it provides a rigorous, principled analysis that generates actionable knowledge for the community.
*   **Value of Ablation Studies:** It underscores the immense value of systematic ablation studies. The findings—such as model scaling being task-dependent and video encoders underperforming image encoders—are non-obvious and challenge common assumptions.
*   **Practical Guidance:** The paper offers clear, practical guidance for researchers and practitioners looking to build similar systems. For instance, the recommendation to use `CEM L2` as a default planner, or to use `Nevergrad` for easier setup, is immediately useful. The insights on matching model capacity to data complexity are also crucial for efficient research.
*   **Critique:** While comprehensive, the analysis could be expanded. For example, the study focuses on frozen encoders. Investigating the effects of fine-tuning the encoder could be a valuable addition. Additionally, the planning cost is a simple distance in latent space. Exploring learned cost functions or integrating language-based goal descriptions could enable more flexible and complex behaviors.
*   **Inspiration:** The work inspires a shift in focus from solely designing new models to deeply understanding the ones we already have. It shows that significant performance gains can be achieved not just through architectural innovation, but through careful and methodical tuning of all parts of the system—training, data, and inference-time optimization. It reinforces the idea that an AI system is more than just its core model; it is an entire ecosystem of interacting components.