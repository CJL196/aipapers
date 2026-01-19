# 1. Bibliographic Information

## 1.1. Title
villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models

## 1.2. Authors
Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Ruiyau Wan, Xi Xia, Zao Jau, Che Jia. The authors are affiliated with prominent research institutions including **Microsoft Research**, **Tsinghua University**, **Wuhan University**, **Hong Kong University of Science and Technology**, and **Nanjing University**.

## 1.3. Journal/Conference
This paper was published on **arXiv** (2025-07-31). Given the affiliations and the depth of the research, it represents a state-of-the-art contribution to the field of robotics and artificial intelligence, likely targeted at top-tier conferences like **CVPR**, **ICLR**, or **CoRL**.

## 1.4. Publication Year
2025

## 1.5. Abstract
Vision-Language-Action (VLA) models are a popular paradigm for learning robot policies that generalize to novel scenarios. This paper introduces `villa-X`, a Vision-Language-Latent-Action (`ViLLA`) framework. It improves how **latent actions** (abstract representations of motion) are learned and integrated into VLA pre-training. By adding a **proprioceptive Forward Dynamics Model (proprio-FDM)**, the framework grounds latent actions in physical dynamics. The authors demonstrate that `villa-X` achieves superior performance in both simulation (SIMPLER benchmark) and real-world tasks involving grippers and dexterous hands, showing strong zero-shot generalization capabilities.

## 1.6. Original Source Link
- **PDF Link:** [https://arxiv.org/pdf/2507.23682v3](https://arxiv.org/pdf/2507.23682v3)
- **Status:** Preprint/Research Paper (v3).

  ---

# 2. Executive Summary

## 2.1. Background & Motivation
The current trend in robotics is to train **Vision-Language-Action (VLA)** models on massive datasets. However, robot-specific data (with labels for exactly what the robot did) is scarce compared to the vast amount of human video data available on the internet. To bridge this gap, researchers use **Latent Actions**.

**What is the core problem?**
1.  **Lack of Physical Grounding:** Existing latent action models learn motion primarily by looking at visual changes (pixels). However, some critical robot movements (like rotating a gripper) might cause tiny pixel changes but are vital for control.
2.  **Inefficient Integration:** Previous methods often used latent actions just to initialize weights or as simple inputs, failing to fully exploit the structured relationship between high-level "intent" (latent actions) and low-level "execution" (robot actions).

**Why is this important?**
If a model doesn't understand the physical constraints and dynamics of a robot, it cannot effectively transfer knowledge from a human video (where a person picks up a cup) to a robot arm.

## 2.2. Main Contributions / Findings
-   **Improved Latent Action Learning:** Introduced `proprio-FDM`, which forces the model to predict future robot states and actions based on latent actions, ensuring they are "physically grounded."
-   **Joint Diffusion Actor:** Developed the `ACT` (Actor) module that jointly models latent and robot actions using a **diffusion-based framework**, allowing the robot to "plan" its latent actions before executing fine-grained movements.
-   **Zero-Shot Generalization:** The model can generate plans for robot embodiments it has never seen before and understand open-vocabulary symbolic icons (e.g., "touch the corn" on a card).
-   **State-of-the-Art Performance:** Outperformed existing baselines in the `SIMPLER` simulation and real-world tests on both standard grippers and 12-Degree-of-Freedom (DoF) dexterous hands.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

-   **Vision-Language-Action (VLA) Model:** A type of AI that takes an image (Vision) and a command (Language) to produce a control signal (Action) for a robot. Think of it as a "GPT for robots."
-   **Latent Action:** Instead of predicting specific joint angles immediately, the model predicts an abstract "intent" or "motion summary." For example, "move toward the cup" is a latent action, while "move joint 1 by 0.5 degrees" is a raw action.
-   **Forward Dynamics Model (FDM):** A model that predicts what the future will look like given the current state and an action. A **Visual FDM** predicts the next image frame; a **Proprioceptive FDM** predicts the next internal state (like joint positions).
-   **Inverse Dynamics Model (IDM):** A model that looks at two frames (before and after) and tries to figure out what action must have happened to get from A to B.
-   **Diffusion Models / Flow Matching:** These are generative AI techniques (similar to those used in Stable Diffusion). They start with random noise and iteratively "refine" it to create a high-quality output—in this case, a smooth sequence of robot actions.

## 3.2. Previous Works
The paper builds on several key concepts:
-   **RT-1 / RT-2:** Early VLA models that showed robots could learn from large-scale data.
-   **LAPA / Moto:** Recent works that started using latent actions to learn from unlabeled videos.
-   **Open X-Embodiment:** A massive dataset combining data from many different types of robots, which is used here for pre-training.

## 3.3. Technological Evolution
In the past, robots were programmed for specific tasks. Then came **Imitation Learning**, where robots copied humans. Now, the field is moving toward **Foundation Models**, where a single model learns from millions of videos (human and robot) to perform almost any task in any environment. `villa-X` represents the next step: making these foundation models understand the "physics" behind the movements, not just the "pixels."

## 3.4. Differentiation Analysis
Unlike previous models like `LAPA` (which only used latent actions for initialization) or `Go-1` (which used discrete tokens), `villa-X` uses a **continuous joint diffusion process**. It also uniquely includes an **embodiment context** ($c_e$), which tells the model "I am currently a Realman arm" or "I am a WidowX robot," helping it adjust its physical expectations accordingly.

---

# 4. Methodology

The `villa-X` framework consists of two primary stages: learning a physically grounded latent action space and then using that space to train a high-performance policy.

## 4.1. Principles
The core intuition is that a latent action should not just describe *how the video changes*, but *how the robot moves*. By supervising the latent actions with internal robot data (proprioception), the model learns a representation that is much more useful for real-world control.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Latent Action Model (LAM)
The `LAM` is responsible for compressing a video clip into a sequence of latent tokens.

**Step 1: Inverse Dynamics Prediction**
The model uses an `Inverse Dynamics Model (IDM)` to extract the latent action $z_t$ from a pair of observations (images) $o_t$ and $o_{t+K}$:
\$
z_t = \mathrm{IDM}(o_t, o_{t+K})
\$
*   $o_t$: The current image frame.
*   $o_{t+K}$: The image frame $K$ steps in the future.
*   $z_t$: The learned latent action token representing the motion between these frames.

**Step 2: Visual Reconstruction**
To ensure the latent action captures visual motion, a `Visual FDM` reconstructs the future frame:
\$
\hat{o}_{t+K} = \mathrm{FDM}(o_t, z_t)
\$
*   $\hat{o}_{t+K}$: The predicted future image.

**Step 3: Proprioceptive Grounding (The Innovation)**
This is the "villa-X" improvement. The authors add a `proprio-FDM` that predicts future internal robot states ($q$) and actions ($a$):
\$
(\hat{q}_{t+1}, ..., \hat{q}_{t+K}, \hat{a}_{t+1}, ..., \hat{a}_{t+K}) = \mathrm{proprio-FDM}(q_t, z_t, c_e)
\$
*   $q_t$: Current proprioceptive state (e.g., joint angles).
*   $c_e$: **Embodiment Context**. This is a vector that identifies the specific robot type and control frequency.
*   **Purpose:** This forces the latent token $z_t$ to encode physical properties, not just pixel shifts.

    The figure below (Figure 1 from the paper) illustrates this grounding process:

    ![Figure 1: (a) A standard Latent Action Model (LAM) learns a latent action `z _ { t }` primarily through visual reconstruction, predicting a future frame $\\hat { o } _ { t + K }$ from the current frame `o _ { t }` and latent action `z _ { t }` (b) Our proposed model enhances this by adding a proprio-FDM. This auxiliary module predicts future robot states $\\widehat { q } _ { t + 1 : t + K }$ and actions $\\hat { a } _ { t : t + K - 1 }$ conditioned on an embodiment context $c _ { e , \\tiny { \\mathscr { C } } }$ enabling the latent actions to be better grounded in physical dynamics.](images/1.jpg)
    *该图像是示意图，展示了传统的潜在动作模型与提出的模型之间的对比。图(a)中，现有潜在动作模型通过视觉重建学习潜在动作 $z_t$，预测未来帧 $\hat{o}_{t+K}$。图(b)则展示了通过添加proprio-FDM模块的改进模型，该模块能够在给定身体上下文 $c_e$ 的条件下预测未来机器人状态 $\hat{q}_{t+1:t+K}$ 和动作 $\hat{a}_{t:t+K-1}$，从而更好地将潜在动作与物理动态结合。*

### 4.2.2. Actor Module (ACT)
Once the `LAM` is trained, the `ACT` module uses these latent actions to perform tasks.

**Step 1: Joint Factorization**
The policy $\pi$ is split into two parts: a high-level latent planner and a low-level robot executor:
\$
\pi(a_{t:t+m-1}, z_{t:t+(n-1)K}^{K} | o_t, l, q_t, c_e) = \pi_{robot}(a | z, o, l, q, c_e) \cdot \pi_{latent}(z | o, l)
\$
*   $\pi_{latent}$: Predicts a sequence of "intentions" (latent actions) based on the image $o_t$ and command $l$.
*   $\pi_{robot}$: Takes those intentions and turns them into actual robot movements $a$.

**Step 2: Flow Matching / Joint Diffusion**
The model uses a technique called **Flow Matching** to generate these actions. It trains a network $v_{\tau}^{\theta}$ to "denoise" random noise into a structured action sequence:
\$
L_{\tau}(\theta) = \mathbb{E}_{p(x_t|O_t), q(x_t^{\tau}|x_t)} \| v_{\tau}^{\theta}(x_t^{\tau}, O_t) - u(x_t^{\tau} | x_t) \|^2
\$
*   $\tau$: The timestep in the diffusion process (from noise to data).
*   $x_t^{\tau}$: The noisy target at step $\tau$.
*   $u(x_t^{\tau} | x_t)$: The "velocity" or direction needed to clean up the noise.
*   $O_t$: The conditioning information (images, language, state).

    The architecture of the `ACT` module is shown in Figure 2:

    ![Figure 2: Architecture of ACT: A hierarchical policy that predicts latent action plans and conditions robot action generation on them, incorporating embodiment context and attention masking.](images/2.jpg)
    *该图像是一个示意图，展示了ACT架构的层次策略，该策略预测潜在动作计划并基于这些计划调节机器人动作生成，同时融入了体现上下文和注意力掩蔽的机制。*

### 4.2.3. Attention Masking Strategy
To prevent the robot from becoming "lazy" and only relying on latent actions (which might be noisy), they use **Stochastic Masking**. During training, $50\%$ of the time, the model is forced to ignore the latent actions and predict the robot actions using only the image and command. This makes the model more robust.

---

# 5. Experimental Setup

## 5.1. Datasets
-   **Robot Data:** 1.6 Million trajectories (223.5 Million frames) from `Open X-Embodiment` and `AgiBot`.
-   **Human Video Data:** 3.6 Million clips from various datasets like `Ego4D` (first-person views of people doing chores), `Epic-Kitchens`, and `Something-Something V2`.
-   **Example Sample:** A command like "pick up the green block" paired with a video of a person's hand or a robot arm performing the task.

## 5.2. Evaluation Metrics

### 5.2.1. Success Rate
1.  **Conceptual Definition:** The percentage of trials where the robot successfully completes the instructed task (e.g., placing a coke can in a drawer).
2.  **Mathematical Formula:**
    \$
    \text{Success Rate} = \frac{N_{success}}{N_{total}} \times 100\%
    \$
3.  **Symbol Explanation:** $N_{success}$ is the count of successful trials; $N_{total}$ is the total number of attempts.

### 5.2.2. L1 Error (for Probing)
1.  **Conceptual Definition:** Used in the "probing" experiment to see how well the latent actions can predict raw robot actions. It measures the absolute distance between the predicted action and the real one.
2.  **Mathematical Formula:**
    \$
    L_1 = \sum_{i=1}^{D} |a_i - \hat{a}_i|
    \$
3.  **Symbol Explanation:** $a_i$ is the ground truth action dimension; $\hat{a}_i$ is the predicted action; $D$ is the number of action dimensions (e.g., 7 or 8).

## 5.3. Baselines
The authors compared `villa-X` against a wide range of state-of-the-art models:
-   **VLA Models:** `RT-1-X`, `Octo-base`, `OpenVLA`, $\pi_0$.
-   **Latent-Action Methods:** `MoTo`, `LAPA`.
-   **World Model Methods:** `GR00T`.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
-   **Physics Grounding Works:** The "probing" experiment (Figure 3) showed that when the `proprio-FDM` module was included (`w/pp`), the latent actions were much better at predicting actual robot actions than when it was omitted (`wo/pp`).
-   **Simulation Dominance:** In the `SIMPLER` benchmark, `villa-X` achieved a $77.7\%$ average success rate on the Google Robot, significantly higher than `OpenVLA` ($32.7\%$) and $\pi_0$ ($58.7\%$).
-   **Real-World Generalization:** The model successfully handled a 12-DoF dexterous hand (Xhand), which is much more complex than standard 1-DoF grippers.

## 6.2. Data Presentation (Tables)

The following are the results from **Table 2** of the original paper, showing success rates on the SIMPLER benchmark:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">Google Robot</th>
<th colspan="5">WidowX Robot</th>
</tr>
<tr>
<th>Pick</th>
<th>Move</th>
<th>Drawer</th>
<th>Avg.</th>
<th>Carrot</th>
<th>Eggplant</th>
<th>Spoon</th>
<th>Cube</th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>RT-1-X *</td>
<td>56.7</td>
<td>31.7</td>
<td>59.7</td>
<td>49.4</td>
<td>4.2</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>1.1</td>
</tr>
<tr>
<td>Octo-base *</td>
<td>17.0</td>
<td>4.2</td>
<td>22.7</td>
<td>14.6</td>
<td>8.3</td>
<td>43.1</td>
<td>12.5</td>
<td>0.0</td>
<td>16.0</td>
</tr>
<tr>
<td>OpenVLA *</td>
<td>16.3</td>
<td>46.2</td>
<td>35.6</td>
<td>32.7</td>
<td>0.0</td>
<td>4.1</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
</tr>
<tr>
<td>π0</td>
<td>72.7</td>
<td>65.3</td>
<td>38.3</td>
<td>58.7</td>
<td>0.0</td>
<td>62.5</td>
<td>29.1</td>
<td>16.6</td>
<td>27.1</td>
</tr>
<tr>
<td><strong>Ours (villa-X)</strong></td>
<td><strong>98.7</strong></td>
<td><strong>75.0</strong></td>
<td><strong>59.3</strong></td>
<td><strong>77.7</strong></td>
<td><strong>46.3</strong></td>
<td><strong>64.6</strong></td>
<td><strong>77.9</strong></td>
<td><strong>61.3</strong></td>
<td><strong>62.5</strong></td>
</tr>
</tbody>
</table>

*Note: "*" indicates the model was evaluated directly without embodiment-specific fine-tuning.*

The following are the results from **Table 9**, showing performance on the `LIBERO` task suites:

<table>
<thead>
<tr>
<th>Method</th>
<th>Spatial</th>
<th>Object</th>
<th>Goal</th>
<th>Long</th>
<th>Average</th>
</tr>
</thead>
<tbody>
<tr>
<td>Diffusion Policy</td>
<td>78.3</td>
<td>92.5</td>
<td>68.3</td>
<td>50.5</td>
<td>72.4</td>
</tr>
<tr>
<td>OpenVLA</td>
<td>84.7</td>
<td>88.4</td>
<td>79.2</td>
<td>53.7</td>
<td>76.5</td>
</tr>
<tr>
<td>π0-FAST</td>
<td>96.4</td>
<td>96.8</td>
<td>88.6</td>
<td>60.2</td>
<td>85.5</td>
</tr>
<tr>
<td><strong>Ours</strong></td>
<td><strong>97.5</strong></td>
<td><strong>97.0</strong></td>
<td><strong>91.5</strong></td>
<td><strong>74.5</strong></td>
<td><strong>90.1</strong></td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The authors removed the "latent-action expert" to see if the model still worked (`Ours w/o latent`). The performance dropped from $77.7\%$ to $36.5\%$ on the Google robot. This proves that the latent action planning is not just a "bonus" but a **core driver** of the model's intelligence.

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
`villa-X` demonstrates that the key to better robot "foundation models" is a **physically grounded latent action space**. By forcing the model to predict internal robot states during the learning of latent actions, it creates a much stronger bridge between human visual data and robot control. The resulting system is more generalizable, powerful, and capable of zero-shot transfer to new robot bodies.

## 7.2. Limitations & Future Work
-   **Planning Strategy:** The current "latent expert" generates plans, but the authors haven't yet implemented a "critic" that can check multiple plans and pick the best one.
-   **Vision Foundation Models:** The authors suggest that future versions could use even larger pre-trained VLMs as the "backbone" to further improve symbolic understanding.
-   **Keypoint Detection:** They mention that structural cues like "hand pose estimation" or "keypoint detection" could eventually replace low-level joint states for even better grounding.

## 7.3. Personal Insights & Critique
-   **Innovation:** The use of an "Embodiment Context" is a very practical solution to the "heterogeneous data" problem (where one robot uses joint angles and another uses end-effector coordinates). It allows the model to learn a shared latent space while acknowledging physical differences.
-   **Transferability:** The zero-shot visualization (Figure 4) showing the model controlling a "Realman" arm it had never seen is extremely impressive. It suggests we are getting closer to a "Universal Robot Controller."
-   **Potential Issue:** One concern with diffusion-based models in robotics is **inference speed**. While the paper mentions `π0-FAST`, the latency of `villa-X` in high-speed, real-time control scenarios remains a point for further investigation.