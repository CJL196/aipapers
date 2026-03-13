# 1. Bibliographic Information

## 1.1. Title
**Title:** Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning

## 1.2. Authors
**Authors:** Yuan Liu, Haoran Li, Shuai Tian, Yuxing Qin, Yuhui Chen, Yupeng Zheng, Yongzhen Huang, Dongbin Zhao.

**Affiliations:** 
1. School of Artificial Intelligence, Beijing Normal University, Beijing, China.
2. Institute of Automation, Chinese Academy of Sciences (CASIA), Beijing, China.
3. School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China.
4. Beijing Academy of Artificial Intelligence, Beijing, China.

## 1.3. Journal/Conference
**Venue:** arXiv Preprint (Submitted to peer review).
**Publication Date:** 2026-02-11T04:05:03.000Z.
**Status:** The paper is listed as an arXiv preprint (identifier: arXiv:2602.10503v1). While not formally assigned to a conference proceeding in this text, its inclusion in arXiv indicates it is an open-access research contribution intended for community review and dissemination prior to or alongside formal publication.

## 1.4. Publication Year
**Year:** 2026.

## 1.5. Abstract
The abstract summarizes the core problem where Supervised Fine-Tuning (SFT) in Vision-Language-Action (VLA) models leads to high data requirements and catastrophic forgetting. To address this, the authors propose **LifeLong-RFT**, a Reinforcement Fine-Tuning (RFT) strategy independent of online environmental feedback. It utilizes a **Multi-Dimensional Process Reward (MDPR)** mechanism consisting of Quantized Action Consistency Reward (QACR), Continuous Trajectory Alignment Reward (CTAR), and Format Compliance Reward (FCR). Experiments show improved performance in multi-task and continual learning settings, achieving significant success rate gains over SFT baselines.

## 1.6. Original Source Link
**Link:** https://arxiv.org/abs/2602.10503
**PDF Link:** https://arxiv.org/pdf/2602.10503v1

---

# 2. Executive Summary

## 2.1. Background & Motivation
**Background:** Vision-Language-Action (VLA) models represent a paradigm shift in robotics, enabling general-purpose policies by mapping visual and linguistic inputs directly to control actions. These models are typically pretrained on large-scale datasets and then adapted to specific robots using Supervised Fine-Tuning (SFT).
**Problem:** The paper identifies two critical limitations of current SFT approaches for adapting VLAs:
1.  **Data Hunger:** SFT requires substantial amounts of task-specific data, hindering rapid adaptation in low-data or few-shot scenarios.
2.  **Catastrophic Forgetting:** As new skills are learned, previously acquired knowledge is often degraded or erased.
    **Motivation:** These challenges prevent VLA models from evolving into "long-lived agents" capable of continually acquiring new skills throughout their deployment. Existing Reinforcement Learning (RL) methods for VLA post-training often rely on expensive environment interactions or unstable reward models. The authors seek a method that enables efficient, continual adaptation without these drawbacks.

## 2.2. Main Contributions / Findings
**Core Contribution:** The paper proposes **LifeLong-RFT**, a reinforcement fine-tuning strategy designed for continual learning in VLA models. It introduces a **Multi-Dimensional Process Reward (MDPR)** mechanism that quantifies action chunk quality across three dimensions without requiring online environmental feedback.
**Key Findings:**
1.  **Effectiveness:** LifeLong-RFT outperforms standard SFT in both multi-task learning and continual learning benchmarks (SimplerEnv, LIBERO, Real-world).
2.  **Continual Learning Gain:** On the LIBERO benchmark, the method achieves a **22% gain** in average success rate over SFT in continual learning settings.
3.  **Data Efficiency:** The model effectively adapts to new tasks using only **20% of the training data** required by traditional baselines.
4.  **Component Validation:** Ablation studies confirm that each component of the MDPR (QACR, CTAR, FCR) is essential for optimal performance.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, several foundational concepts must be defined:
*   **Vision-Language-Action (VLA) Models:** These are robotic policy models that take multimodal observations (images/video) and natural language instructions as input, outputting robot control actions (e.g., gripper states, end-effector pose). They aim to unify perception and decision-making in a single neural network.
*   **Supervised Fine-Tuning (SFT):** A post-training technique where a pretrained model is updated using labeled data specific to a downstream task. In robotics, this means providing examples of state-action pairs `(o, a)` and minimizing the difference between predicted and actual actions.
*   **Catastrophic Forgetting:** A phenomenon in machine learning where a neural network trained on a new task loses the ability to perform previous tasks because the weight updates for the new task overwrite the patterns associated with old tasks.
*   **On-Policy Reinforcement Learning (RL):** An RL paradigm where the agent learns from actions it generates according to its *current* policy (distribution), rather than relying on historical data from a different policy. This allows for self-improvement based on immediate performance feedback.
*   **Chunking-Level Actions:** VLA models often predict actions in sequences or "chunks" (e.g., a series of poses over the next 1 second) rather than single time steps. Optimizing over chunks allows for better long-horizon planning.
*   **Group Relative Policy Optimization (GRPO):** An algorithm used in this paper instead of PPO. Instead of using a separate critic network to estimate value, GRPO compares a group of sampled outputs to calculate advantage estimates, reducing computational overhead.

## 3.2. Previous Works
The paper situates itself within three categories of related work:
1.  **VLA Post-Training:** Traditional methods rely heavily on SFT. Recent works explore RL but often depend on simulators or world models which introduce simulation-to-reality gaps or reward hacking risks.
2.  **Reinforcement Fine-Tuning:** Existing approaches include simulation-based methods (privileged info), real-world strategies (high human cost), and world-model-driven methods (prediction errors). Most lack focus on *continual* learning properties.
3.  **Continual Learning in Robotics:** Prior techniques include parameter isolation (task-specific adapters) or memory replay. In foundation models, merging expert models (MergeVLA) or knowledge distillation (Stellar VLA) are common attempts, but scaling to massive VLA settings remains difficult.

## 3.3. Technological Evolution
The field has evolved from hierarchical robot controllers to end-to-end VLA models. Initially, VLA models were adapted via imitation learning (SFT). Recognizing SFT's fragility, researchers began applying RL. However, most RL methods focused on improving single-task performance rather than *lifelong* adaptation. This paper represents an evolution towards integrating RL specifically for *continual* capabilities, removing the dependency on environment interaction during training to reduce costs and instability.

## 3.4. Differentiation Analysis
Unlike existing RFT methods that require online interaction or world models, **LifeLong-RFT** is distinct because:
1.  **Offline Training:** It performs fine-tuning offline using batch demonstrations without needing real-time environment feedback.
2.  **Process Rewards:** Instead of outcome-only rewards (success/fail), it uses process rewards (MDPR) evaluated at intermediate action chunks.
3.  **Continual Focus:** It explicitly targets the balance between plasticity (learning new tasks) and stability (retaining old tasks), whereas many baselines optimize solely for the current task's success rate.

    ---

# 4. Methodology

## 4.1. Principles
The core idea behind **LifeLong-RFT** is to leverage the robustness of on-policy reinforcement learning against catastrophic forgetting, combined with a carefully designed reward system that mimics environment feedback without actually interacting with it. The method decomposes the complex task of robotic manipulation into discrete tokens and continuous trajectories, rewarding consistency, alignment, and format compliance separately.

The following figure (Figure 2 from the original paper) illustrates the optimization strategy using the Multi-Dimensional Process Reward mechanism:

![algorithm with the Multi-Dimensional Process Reward mechanism to facilitate policy optimization.](images/2.jpg)
*该图像是示意图，展示了使用多维过程奖励机制优化强化学习的策略。图中包含了 VLA 模型的观察和指令输入，涵盖了 QACR、CTAR 和 FCR 三个奖励机制，用于确保精确的行为预测、对齐连续动作和格式合规性。*

## 4.2. Core Methodology In-depth

### 4.2.1. Chunking-Level On-Policy Reinforcement Learning
The method employs **Group Relative Policy Optimization (GRPO)**. Unlike standard Proximal Policy Optimization (PPO) which requires a critic network to estimate values, GRPO calculates advantages by comparing a group of generated actions.

For a given observation $o$ and language instruction $l$, the model samples a group of $G$ action outputs $\{\mathbf{a}_i\}_{i=1}^G$ from the old policy $\pi_{\theta_{\mathrm{old}}}(\mathbf{a}|o, l)$.

The reward $r_i$ for each output is computed via the proposed MDPR mechanism. The relative advantage $A_i$ for each output is calculated based on the mean and standard deviation of the intra-group rewards:

\$
A _ { i } = { \frac { r _ { i } - \operatorname* { m e a n } ( \{ r _ { 1 } , \dots , r _ { G } \} ) } { \operatorname { s t d } ( \{ r _ { 1 } , \dots , r _ { G } \} ) } } .
\$

Here, $r_i$ is the scalar reward for sample $i$. $\operatorname*{mean}$ calculates the average reward of the group, and $\operatorname{std}$ calculates the standard deviation. This normalization ensures stable gradient updates regardless of the absolute scale of the reward.

Once the advantage $A_i$ is determined, the policy parameters $\theta$ are optimized by maximizing the following objective function:

\$
\begin{array} { l } { { \displaystyle { \cal J } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { ( o , l ) \sim \mathcal { B } , \{ { \bf a } _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \vert o , l ) } } } \\ { { \displaystyle ~ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \lbrace \operatorname* { m i n } \lbrack \frac { \pi _ { \theta } \left( { \bf a } _ { i } \vert o , l \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( { \bf a } _ { i } \vert o , l \right) } A _ { i } , } } \\ { { \displaystyle ~ \mathrm { c l i p } \left( \frac { \pi _ { \theta } \left( { \bf a } _ { i } \vert o , l \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( { \bf a } _ { i } \vert o , l \right) } , 1 - \epsilon , 1 + \epsilon \right) A _ { i } \rbrack } } \\ { { \displaystyle ~ - \gamma D _ { K L } \left[ \pi _ { \theta } \vert \vert \pi _ { \mathrm { r e f } } \vert \right. } , } } \end{array}
\$

Where:
*   $\mathcal{B}$ denotes the dataset of expert demonstrations containing observations $o$ and instructions $l$.
*   $\pi_{\theta}$ is the current policy being updated.
*   $\pi_{\theta_{\mathrm{old}}}$ is the policy before the update step.
*   $\frac{\pi_{\theta}(\mathbf{a}_i|o,l)}{\pi_{\theta_{\mathrm{old}}}(\mathbf{a}_i|o,l)}$ is the probability ratio of the action under the new vs. old policy.
*   $\epsilon$ is a clipping parameter that constrains how much the policy can change (typical values like 0.2).
*   $\gamma$ modulates the strength of the KL divergence regularization term $D_{KL}[\pi_\theta || \pi_{\mathrm{ref}}]$.
*   $\pi_{\mathrm{ref}}$ is a reference policy (usually the initialized or base model) used to prevent the new policy from deviating too drastically, ensuring stability.

### 4.2.2. Multi-Dimensional Process Reward (MDPR)
To provide the reward $r_i$ without environment interaction, the authors design the MDPR. This decomposes the evaluation of action chunks into three complementary dimensions.

#### 1) Quantized Action Consistency Reward (QACR)
Since VLA models generate discrete action tokens, QACR assesses the consistency between generated tokens and ground truth tokens.

First, a format check verifies compliance with the tokenizer specifications (action chunk size, dimensions). Invalid generations receive a zero reward. If valid, the consistency is measured position-wise matching between the predicted token sequence $\mathbf{a}=\{a_u\}_{u=1}^U$ and ground truth $\tilde{\mathbf{a}}=\{\tilde{a}_v\}_{v=1}^V$.

The formula for QACR is:

\$
\mathrm { Q A C R } = \left\{ \begin{array} { l l } { \displaystyle \frac { \sum _ { \ell = 1 } ^ { \operatorname* { m i n } ( U , V ) } \mathbb { I } ( a _ { \ell } = \tilde { a } _ { \ell } ) } { \operatorname* { m a x } ( U , V ) } , } & { \mathrm { i f ~ v a l i d } } \\ { \displaystyle 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.
\$

Where:
*   $\mathbb{I}(\cdot)$ is an indicator function returning 1 if tokens match, 0 otherwise.
*   The denominator normalizes by the maximum length of the sequences.
*   This ensures the model predicts the correct discrete tokens accurately.

#### 2) Continuous Trajectory Alignment Reward (CTAR)
Physical robots execute continuous trajectories. CTAR aligns decoded continuous action chunks with reference trajectories. It uses the Fast+ tokenizer to decode tokens $\mathbf{a}$ into continuous action vectors $\mathbf{y}$ (pose and gripper).

The calculation involves decoding both predicted and ground truth actions into $\mathbf{y}$ and $\tilde{\mathbf{y}}$. The reward sums pose and grip rewards over the action chunk size $H$.

The formula for CTAR is:

\$
\mathrm { C T A R } = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { H } \sum _ { t = 1 } ^ { H } \left( \beta \cdot r _ { t } ^ { \mathrm { p o s e } } + \left( 1 - \beta \right) \cdot r _ { t } ^ { \mathrm { g r i p } } \right) , } & { \mathrm { i f ~ v a l i d } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.
\$

Where:
*   $r_{t}^{\mathrm{pose}} = \exp(-\alpha \cdot d_{t})$, with $d_{t}$ being the normalized L1 distance between predicted and ground truth pose vectors. $\alpha$ controls sensitivity to pose deviation.
*   $r_{t}^{\mathrm{grip}} = \mathbb{I}(\mathbf{y}_{t}^{\mathrm{grip}} = \tilde{\mathbf{y}}_{t}^{\mathrm{grip}})$, a binary reward for gripper state accuracy.
*   $\beta$ balances the importance of pose vs. gripper rewards.

#### 3) Format Compliance Reward (FCR)
FCR ensures structural validity. It is a binary reward that returns 1 if the model output adheres to the predefined output format (decodable by the tokenizer), and 0 otherwise.

\$
\mathrm { F C R } = \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } { \mathrm { v a l i d } } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right.
\$

### 4.2.3. Synthesis of MDPR
Finally, the total reward $r_i$ for the GRPO objective is synthesized from the three components using weighted summation:

\$
{ \bf M D P R } = \boldsymbol { \omega } \cdot { \bf Q A C R } + ( 1 - \boldsymbol { \omega } ) \cdot { \bf C T A R } + \boldsymbol { \lambda } \cdot { \bf F C R } ,
\$

Where:
*   $\omega \in [0, 1]$ governs the trade-off between discrete action consistency (QACR) and continuous trajectory alignment (CTAR).
*   $\lambda$ scales the significance of structural format compliance (FCR).

    The combination of these elements forms the core of the LifeLong-RFT algorithm, allowing offline policy optimization grounded in task-specific constraints rather than external environment signals.

---

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize three primary environments to validate performance across simulation and reality:
1.  **SimplerEnv:** A simulation environment for evaluating visual matching tasks on WidowX and Google Robot platforms. Data comes from BridgeData V2 (WidowX) and Fractal (Google Robot).
2.  **LIBERO:** A benchmark specifically designed for lifelong robot learning. It contains task suites (Object, Spatial, Goal, Long) with third-person and wrist camera inputs. Each suite comprises 10 tasks.
3.  **Real-World (Franka Robot):** Physical experiments conducted on a Franka Emika Panda manipulator involving four tasks: Pick Banana, Pick Bread, Pull Drawer, and Hang Chinese Knot.

    **Data Example:** In LIBERO, a task might involve an instruction like "Put the apple into the basket," paired with video demonstrations showing the arm moving to grasp the apple and place it in the target zone. The data format includes RGB images, language embeddings, and corresponding action trajectories.

## 5.2. Evaluation Metrics
Three key metrics are used to assess continual learning capabilities, derived from task success rates:
1.  **Forward Transfer (FWT):** Measures how well the model adapts to *new* tasks after learning previous ones. Higher is better.
    Formula:
    $$
    \text{FWT} = \sum_{k \in [K]} \frac{s_{k,k}}{K}
    $$
    Where $s_{k,k}$ is the success rate on the current task $k$ after learning $k$ tasks.
2.  **Negative Backward Transfer (NBT):** Measures catastrophic forgetting. Lower values indicate better retention of previous skills.
    Formula:
    $$
    \text{NBT}_k = \frac{1}{K-k}\sum_{q=k+1}^{K} (s_{k,k} - s_{q,k})
    $$
    $$
    \text{NBT} = \sum_{k \in [K]} \frac{\text{NBT}_k}{K}
    $$
    Where $s_{q,k}$ is the success rate on task $k$ after learning up to task $q$.
3.  **Area Under the Success Rate Curve (AUC):** Reflects the average performance across all tasks over the entire learning timeline. Higher is better.
    Formula:
    $$
    \text{AUC}_k = \frac{1}{K-k+1} (s_{k,k} + \sum_{q=k+1}^{K} s_{q,k})
    $$
    $$
    \text{AUC} = \sum_{k \in [K]} \frac{\text{AUC}_k}{K}
    $$

## 5.3. Baselines
The proposed method is compared against various state-of-the-art models including:
*   **Continuous Action Models:** Octo-Base, GROO TN1, $\pi_0$, OpenVLA-OFT.
*   **Discrete Action Models:** TraceVLA, RT-1-X, OpenVLA, SpatialVLA, $\pi_0$-FAST.
*   **Other Fine-Tuning Methods:** ThinkAct, NORA-1.5 (with DPO).
*   **Continual Learning Methods:** BUDS, LOTUS, SPECI.

    These baselines represent a mix of standard SFT adaptations and recent reinforcement fine-tuning approaches to establish a comprehensive performance comparison.

---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results demonstrate that LifeLong-RFT consistently outperforms Supervised Fine-Tuning (SFT) baselines.

**Simulation Performance:**
In SimplerEnv, the method improves average success rates by **3.5%** on WidowX and **4.4%** on Google Robot compared to SFT. On the more challenging LIBERO benchmark, it achieves a superior average success rate of **95.6%**.

The following are the results from **Table I** of the original paper (Multi-Task learning performance on SimplerEnv):

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Training Strategy</th>
<th colspan="5">WidowX (Visual Matching)</th>
<th colspan="4">Google Robot (Visual Matching)</th>
</tr>
<tr>
<th>Put Carrot on Plate</th>
<th>Stack Blocks</th>
<th>Put Spoon on Towel</th>
<th>Put Eggplant in Basket</th>
<th>Avg</th>
<th>Pick Coke Can</th>
<th>Move Near</th>
<th>Open/Close Drawer</th>
<th>Avg</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="11"><b>Continuous Action Models</b></td>
</tr>
<tr>
<td>Octo-Base [66]</td>
<td>SFT</td>
<td>8.3</td>
<td>0.0</td>
<td>12.5</td>
<td>43.1</td>
<td>16.0</td>
<td>17.0</td>
<td>4.2</td>
<td>22.7</td>
<td>16.8</td>
</tr>
<tr>
<td>RoboVLM [39]</td>
<td>SFT</td>
<td>25.0</td>
<td>12.5</td>
<td>29.2</td>
<td>58.3</td>
<td>31.3</td>
<td>77.3</td>
<td>61.7</td>
<td>43.5</td>
<td>63.4</td>
</tr>
<tr>
<td>GROOT N1.5 [53]</td>
<td>SFT</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>69.3</td>
<td>68.7</td>
<td>35.8</td>
<td>52.4</td>
</tr>
<tr>
<td>$\pi_{0.5}$ [26]</td>
<td>SFT</td>
<td>58.8</td>
<td>21.3</td>
<td>63.3</td>
<td>79.2</td>
<td>55.7</td>
<td>72.7</td>
<td>65.3</td>
<td>38.3</td>
<td>58.7</td>
</tr>
<tr>
<td>ThinkAct [22]</td>
<td>SFT + RFT</td>
<td>37.5</td>
<td>8.7</td>
<td>58.3</td>
<td>70.8</td>
<td>43.8</td>
<td>92.0</td>
<td>72.4</td>
<td>50.0</td>
<td>71.5</td>
</tr>
<tr>
<td>NORA-1.5 [24]</td>
<td>SFT</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>92.8</td>
<td>78.7</td>
<td>62.2</td>
<td>77.9</td>
</tr>
<tr>
<td>NORA-1.5 [24] (DPO)</td>
<td>SFT+RFT</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>94.0</td>
<td>88.0</td>
<td>66.4</td>
<td>82.8</td>
</tr>
<tr>
<td colspan="11"><b>Discrete Action Models</b></td>
</tr>
<tr>
<td>TraceVLA [80]</td>
<td>SFT</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>28.0</td>
<td>53.7</td>
<td>57.0</td>
<td>42.0</td>
</tr>
<tr>
<td>RT-1-X [7]</td>
<td>SFT</td>
<td>4.2</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>1.1</td>
<td>56.7</td>
<td>31.7</td>
<td>59.7</td>
<td>53.4</td>
</tr>
<tr>
<td>OpenVLA [28]</td>
<td>SFT</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>4.1</td>
<td>1.0</td>
<td>16.3</td>
<td>46.2</td>
<td>35.6</td>
<td>27.7</td>
</tr>
<tr>
<td>SpatialVLA [57]</td>
<td>SFT</td>
<td>25.0</td>
<td>29.2</td>
<td>16.7</td>
<td>100.0</td>
<td>42.7</td>
<td>86.0</td>
<td>77.9</td>
<td>57.4</td>
<td>73.7</td>
</tr>
<tr>
<td>$\pi_0$-FAST [56]</td>
<td>SFT</td>
<td>22.0</td>
<td>83.0</td>
<td>29.0</td>
<td>48.0</td>
<td>45.5</td>
<td>75.3</td>
<td>67.5</td>
<td>42.6</td>
<td>61.9</td>
</tr>
<tr>
<td>NORA-1.5-FAST [24]</td>
<td>SFT</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>88.6</td>
<td>86.4</td>
<td>41.2</td>
<td>72.1</td>
</tr>
<tr>
<td>NORA-Long [25] (Baseline)</td>
<td>SFT</td>
<td>46.0</td>
<td>60.3</td>
<td>80.2</td>
<td>75.7</td>
<td>65.5</td>
<td>86.0</td>
<td>82.3</td>
<td>56.0</td>
<td>74.7</td>
</tr>
<tr>
<td>NORA-Long [25]</td>
<td>RFT (Ours)</td>
<td>50.2</td>
<td>64.4</td>
<td>84.3</td>
<td>77.0</td>
<td>69.0</td>
<td>94.0</td>
<td>84.7</td>
<td>58.5</td>
<td>79.1</td>
</tr>
<tr>
<td>$\Delta$</td>
<td></td>
<td>+4.2</td>
<td>+4.1</td>
<td>+4.1</td>
<td>+1.3</td>
<td>+3.5</td>
<td>+8.0</td>
<td>+2.4</td>
<td>+2.5</td>
<td>+4.4</td>
</tr>
</tbody>
</table>

The following are the results from **Table II** of the original paper (Multi-Task learning performance on LIBERO):

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Training Strategy</th>
<th colspan="4">LIBERO</th>
<th rowspan="2">Avg</th>
</tr>
<tr>
<th>Object</th>
<th>Spatial</th>
<th>Goal</th>
<th>Long</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7"><b>Continuous Action Models</b></td>
</tr>
<tr>
<td>Octo-Base [66]</td>
<td>SFT</td>
<td>85.7</td>
<td>78.9</td>
<td>84.6</td>
<td>51.1</td>
<td>75.1</td>
</tr>
<tr>
<td>GROO TN1 [53]</td>
<td>SFT</td>
<td>97.6</td>
<td>94.4</td>
<td>93.0</td>
<td>90.6</td>
<td>93.9</td>
</tr>
<tr>
<td>$\pi_0$ [6]</td>
<td>SFT</td>
<td>98.8</td>
<td>96.8</td>
<td>95.8</td>
<td>85.2</td>
<td>94.2</td>
</tr>
<tr>
<td>OpenVLA-OFT [29]</td>
<td>SFT</td>
<td>98.1</td>
<td>96.9</td>
<td>95.5</td>
<td>91.1</td>
<td>95.4</td>
</tr>
<tr>
<td>ThinkAct [22]</td>
<td>SFT + RFT</td>
<td>91.4</td>
<td>88.3</td>
<td>87.1</td>
<td>70.9</td>
<td>84.4</td>
</tr>
<tr>
<td>VLA-RFT [35]</td>
<td>SFT + RFT</td>
<td>94.4</td>
<td>94.4</td>
<td>95.4</td>
<td>80.2</td>
<td>91.1</td>
</tr>
<tr>
<td>NORA-1.5 [24]</td>
<td>SFT</td>
<td>96.4</td>
<td>97.3</td>
<td>94.5</td>
<td>89.6</td>
<td>94.5</td>
</tr>
<tr>
<td>NORA-1.5 [24] (DPO)</td>
<td>SFT + RFT</td>
<td>96.0</td>
<td>98.0</td>
<td>95.4</td>
<td>90.5</td>
<td>95.0</td>
</tr>
<tr>
<td colspan="7"><b>Discrete Action Models</b></td>
</tr>
<tr>
<td>TraceVLA [80]</td>
<td>SFT</td>
<td>85.2</td>
<td>84.6</td>
<td>-</td>
<td>75.1</td>
<td>74.8</td>
</tr>
<tr>
<td>OpenVLA [28]</td>
<td>SFT</td>
<td>88.4</td>
<td>84.7</td>
<td>79.2</td>
<td>53.7</td>
<td>76.5</td>
</tr>
<tr>
<td>SpatialVLA [57]</td>
<td>SFT</td>
<td>89.9</td>
<td>88.2</td>
<td>78.6</td>
<td>55.5</td>
<td>78.1</td>
</tr>
<tr>
<td>CoT-VLA [78]</td>
<td>SFT</td>
<td>91.6</td>
<td>87.5</td>
<td>87.6</td>
<td>69.0</td>
<td>83.9</td>
</tr>
<tr>
<td>WorldVLA [8]</td>
<td>SFT</td>
<td>96.2</td>
<td>87.6</td>
<td>83.4</td>
<td>60.0</td>
<td>79.1</td>
</tr>
<tr>
<td>$\pi_0$-Fast [56]</td>
<td>SFT</td>
<td>96.8</td>
<td>96.4</td>
<td>88.6</td>
<td>60.2</td>
<td>85.5</td>
</tr>
<tr>
<td>MolmoAct-7B-D [32]</td>
<td>SFT</td>
<td>95.4</td>
<td>87.0</td>
<td>87.6</td>
<td>77.2</td>
<td>86.6</td>
</tr>
<tr>
<td>TGRPO [15]</td>
<td>SFT + RFT</td>
<td>92.2</td>
<td>90.4</td>
<td>81.0</td>
<td>59.2</td>
<td>80.7</td>
</tr>
<tr>
<td>NORA-Long [25] (Baseline)</td>
<td>SFT</td>
<td>97.5</td>
<td>96.4</td>
<td>91.0</td>
<td>82.4</td>
<td>91.8</td>
</tr>
<tr>
<td>NORA-Long [25]</td>
<td>RFT (Ours)</td>
<td>99.2</td>
<td>98.2</td>
<td>95.8</td>
<td>89.0</td>
<td>95.6</td>
</tr>
<tr>
<td>$\Delta$</td>
<td></td>
<td>+1.7</td>
<td>+1.8</td>
<td>+4.8</td>
<td>+6.6</td>
<td>+3.8</td>
</tr>
</tbody>
</table>

**Real-World Performance:**
On the Franka robot, LifeLong-RFT achieved an average success rate improvement of **8.7%** over the SFT baseline. The following are the results from **Table III**:

<table>
<thead>
<tr>
<th rowspan="2">Task Split</th>
<th>$\pi_0$ [6]</th>
<th>OpenVLA [28]</th>
<th colspan="3">NORA-Long [24]</th>
</tr>
<tr>
<th>SFT</th>
<th>SFT</th>
<th>SFT</th>
<th>RFT (Ours)</th>
<th>$\Delta$</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pick Banana</td>
<td>90.0</td>
<td>75.0</td>
<td>85.0</td>
<td>90.0</td>
<td>+5.0</td>
</tr>
<tr>
<td>Pick Bread</td>
<td>75.0</td>
<td>70.0</td>
<td>75.0</td>
<td>85.0</td>
<td>+10.0</td>
</tr>
<tr>
<td>Pull Drawer</td>
<td>95.0</td>
<td>85.0</td>
<td>95.0</td>
<td>100.0</td>
<td>+5.0</td>
</tr>
<tr>
<td>Hang Chinese Knot</td>
<td>65.0</td>
<td>55.0</td>
<td>60.0</td>
<td>75.0</td>
<td>+15.0</td>
</tr>
<tr>
<td>Overall</td>
<td>81.3</td>
<td>71.3</td>
<td>78.8</td>
<td>87.5</td>
<td>+8.7</td>
</tr>
</tbody>
</table>

### 6.1.1. Continual Learning Results
The most significant validation lies in continual learning. LifeLong-RFT excels at retaining old skills while learning new ones.

The following are the results from **Table IV** of the original paper (Continual learning performance on LIBERO):

<table>
<thead>
<tr>
<th rowspan="2">Task Split</th>
<th rowspan="2">Metrics</th>
<th>BUDS [82]</th>
<th>LOTUS [68]</th>
<th>SPECI [72]</th>
<th>$\pi_0$ [6]</th>
<th>OpenVLA [28]</th>
<th>OpenVLA-OFT [29]</th>
<th colspan="3">NORA-Long [25]</th>
</tr>
<tr>
<th>BC</th>
<th>BC</th>
<th>BC</th>
<th>SFT</th>
<th>SFT</th>
<th>SFT</th>
<th>SFT</th>
<th>RFT (Ours)</th>
<th>$\Delta$</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">LIBERO-Object</td>
<td>FWT ($\uparrow$)</td>
<td>52.0</td>
<td>74.0</td>
<td>83.0</td>
<td>73.0</td>
<td>59.4</td>
<td>89.8</td>
<td>84.8</td>
<td>96.0</td>
<td>+11.2</td>
</tr>
<tr>
<td>NBT ($\downarrow$)</td>
<td>21.0</td>
<td>11.0</td>
<td>10.0</td>
<td>16.2</td>
<td>17.9</td>
<td>3.1</td>
<td>6.8</td>
<td>1.5</td>
<td>-5.3</td>
</tr>
<tr>
<td>AUC ($\uparrow$)</td>
<td>47.0</td>
<td>65.0</td>
<td>78.0</td>
<td>59.3</td>
<td>45.1</td>
<td>87.4</td>
<td>79.7</td>
<td>94.8</td>
<td>+15.1</td>
</tr>
<tr>
<td rowspan="3">LIBERO-Spatial</td>
<td>FWT ($\uparrow$)</td>
<td>-</td>
<td>-</td>
<td>67.0</td>
<td>74.4</td>
<td>64.2</td>
<td>88.6</td>
<td>82.8</td>
<td>94.0</td>
<td>+11.2</td>
</tr>
<tr>
<td>NBT ($\downarrow$)</td>
<td>-</td>
<td>-</td>
<td>6.0</td>
<td>23.7</td>
<td>17.6</td>
<td>9.4</td>
<td>14.0</td>
<td>3.7</td>
<td>-10.3</td>
</tr>
<tr>
<td>AUC ($\uparrow$)</td>
<td>-</td>
<td>-</td>
<td>66.0</td>
<td>55.5</td>
<td>50.8</td>
<td>81.7</td>
<td>71.7</td>
<td>91.2</td>
<td>+19.5</td>
</tr>
<tr>
<td rowspan="3">LIBERO-Goal</td>
<td>FWT ($\uparrow$)</td>
<td>50.0</td>
<td>61.0</td>
<td>74.0</td>
<td>74.6</td>
<td>58.6</td>
<td>90.2</td>
<td>72.8</td>
<td>92.4</td>
<td>+19.6</td>
</tr>
<tr>
<td>NBT ($\downarrow$)</td>
<td>39.0</td>
<td>30.0</td>
<td>20.0</td>
<td>23.9</td>
<td>5.8</td>
<td>13.8</td>
<td>25.2</td>
<td>3.1</td>
<td>-22.1</td>
</tr>
<tr>
<td>AUC ($\uparrow$)</td>
<td>42.0</td>
<td>56.0</td>
<td>65.0</td>
<td>56.3</td>
<td>53.5</td>
<td>79.2</td>
<td>54.4</td>
<td>90.3</td>
<td>+35.9</td>
</tr>
<tr>
<td rowspan="3">LIBERO-Long</td>
<td>FWT ($\uparrow$)</td>
<td>-</td>
<td>-</td>
<td>58.0</td>
<td>53.8</td>
<td>32.0</td>
<td>64.0</td>
<td>61.0</td>
<td>74.2</td>
<td>+13.2</td>
</tr>
<tr>
<td>NBT ($\downarrow$)</td>
<td>-</td>
<td>-</td>
<td>21.0</td>
<td>14.2</td>
<td>14.1</td>
<td>31.4</td>
<td>17.3</td>
<td>12.8</td>
<td>-4.5</td>
</tr>
<tr>
<td>AUC ($\uparrow$)</td>
<td>-</td>
<td>-</td>
<td>46.0</td>
<td>42.5</td>
<td>20.8</td>
<td>38.7</td>
<td>47.3</td>
<td>64.5</td>
<td>+17.2</td>
</tr>
</tbody>
</table>

The following are the results from **Table V** of the original paper (Continual learning performance on real-world):

<table>
<thead>
<tr>
<th rowspan="2">Task Split</th>
<th rowspan="2">Metrics</th>
<th>$\pi_0$ [6]</th>
<th>OpenVLA [28]</th>
<th colspan="3">NORA-Long [25]</th>
</tr>
<tr>
<th>SFT</th>
<th>SFT</th>
<th>SFT</th>
<th>RFT (Ours)</th>
<th>$\Delta$</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Real-World</td>
<td>FWT ($\uparrow$)</td>
<td>58.8</td>
<td>46.3</td>
<td>56.3</td>
<td>80.0</td>
<td>+23.7</td>
</tr>
<tr>
<td>NBT ($\downarrow$)</td>
<td>16.3</td>
<td>17.8</td>
<td>18.3</td>
<td>6.1</td>
<td>-12.2</td>
</tr>
<tr>
<td>AUC ($\uparrow$)</td>
<td>47.9</td>
<td>35.1</td>
<td>44.2</td>
<td>75.9</td>
<td>+31.7</td>
</tr>
</tbody>
</table>

Analysis of these tables reveals that LifeLong-RFT significantly reduces Negative Backward Transfer (NBT), indicating less forgetting. On LIBERO-Goal, it shows a **35.9 point gain** in AUC compared to the SFT baseline, demonstrating exceptional stability when handling sequential goals.

### 6.1.2. Adaptation Efficiency
The method also excels at few-shot adaptation. Figure 4 (from the original paper) compares the number of episodes required to reach high success rates:

![Fig. 4: Adaptation efficiency on representative new tasks.](images/4.jpg)
*该图像是一个图表，展示了不同新任务的适应效率，包括对象新任务、空间新任务、目标新任务和长任务。图中展示了 RFT 和 SFT 在不同训练轮次下的成功率对比，RFT 在大多数任务上表现出更优的成功率，尤其是在训练轮次增多时，明确展示了 LifeLong-RFT 方法的有效性。*

This chart demonstrates that LifeLong-RFT achieves comparable or better performance with far fewer demonstrations (e.g., 5 demos vs 50 demos) than SFT.

### 6.2. Ablation Studies
The authors conduct ablation studies to verify the MDPR components.

The following are the results from **Table VI** of the original paper (Ablation of Multi-Dimensional Process Rewards):

<table>
<thead>
<tr>
<th rowspan="2">Settings</th>
<th colspan="2">Object</th>
<th colspan="2">Spatial</th>
<th colspan="2">Goal</th>
<th colspan="2">Long</th>
<th colspan="2">Avg</th>
</tr>
<tr>
<th>SR</th>
<th>$\Delta$</th>
<th>SR</th>
<th>$\Delta$</th>
<th>SR</th>
<th>$\Delta$</th>
<th>SR</th>
<th>$\Delta$</th>
<th>SR</th>
<th>$\Delta$</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o QACR</td>
<td>97.0</td>
<td>-2.2</td>
<td>96.4</td>
<td>-1.8</td>
<td>92.2</td>
<td>-3.6</td>
<td>85.6</td>
<td>-3.4</td>
<td>92.8</td>
<td>-2.8</td>
</tr>
<tr>
<td>w/o CTAR</td>
<td>8.0</td>
<td>-91.2</td>
<td>6.2</td>
<td>-92.0</td>
<td>2.4</td>
<td>-93.4</td>
<td>2.0</td>
<td>-87.0</td>
<td>4.7</td>
<td>-90.9</td>
</tr>
<tr>
<td>w/o FCR</td>
<td>98.0</td>
<td>-1.2</td>
<td>96.2</td>
<td>-2.0</td>
<td>93.2</td>
<td>-2.6</td>
<td>84.6</td>
<td>-4.4</td>
<td>93.0</td>
<td>-2.6</td>
</tr>
<tr>
<td>RFT (Ours)</td>
<td>99.2</td>
<td>-</td>
<td>98.2</td>
<td>-</td>
<td>95.8</td>
<td>-</td>
<td>89.0</td>
<td>-</td>
<td>95.6</td>
<td>-</td>
</tr>
</tbody>
</table>

**Analysis:** Removing **CTAR** causes a catastrophic drop (~90%) in success rate, confirming that continuous trajectory alignment is the most critical component for physical execution. QACR and FCR provide smaller but necessary refinements for token accuracy and format validity.

Parameter sensitivity analysis (Figure 5 from the original paper) shows the method is robust to variations in weights $\omega$ and $\lambda$:

![Fig. 5: Ablation study on the reward combination weights.](images/5.jpg)
*该图像是条形图，展示了在不同权重组合下的平均成功率。图中分为两部分：(a) 显示了不同 $\omega$ 值（0.1、0.3、0.7、0.9）对平均成功率的影响，成功率从94.6%到95.8%；(b) 显示了不同 $\lambda$ 值（0.1、0.3、0.7、1.0）的影响，成功率在93.2%到95.8%之间。数据可视化帮助分析奖励组合对性能的作用。*

Performance remains high across reasonable weight configurations, with optimal performance observed at $\omega = 0.7$ and $\lambda = 0.1$.

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper presents **LifeLong-RFT**, a robust reinforcement fine-tuning strategy designed to overcome the limitations of Supervised Fine-Tuning (SFT) in Vision-Language-Action (VLA) models. By integrating chunking-level on-policy reinforcement learning with a novel Multi-Dimensional Process Reward (MDPR) mechanism, the method enables VLAs to continually learn new tasks while preserving previously acquired knowledge. The approach eliminates the need for online environmental feedback during training, reducing costs and mitigating safety risks. Comprehensive experiments across simulated and real-world environments demonstrate significant improvements in success rates, forward transfer, and data efficiency compared to existing baselines.

## 7.2. Limitations & Future Work
**Limitations:**
1.  **Action Space Discreteness:** The work primarily focuses on discrete action models. The authors acknowledge that discrete actions currently fall short of the performance levels achievable by continuous action models.
2.  **Deformable Objects:** While successful on rigid objects (Banana, Bread, Drawer), the "Hang Chinese Knot" task (deformable object) showed lower success rates (60%), highlighting challenges with highly deformable items.
3.  **Computational Cost:** Although reduced compared to simulation-based RL, on-policy RL still incurs higher computation than simple SFT.

**Future Work:**
The authors suggest extending the LifeLong-RFT training strategy to **continuous action models** to accelerate industrial application. Further research is needed to improve handling of deformable objects and long-horizon tasks with extremely limited demonstrations.

## 7.3. Personal Insights & Critique
This paper addresses a pivotal bottleneck in embodied AI: the transition from static, one-off adaptation to dynamic, lifelong learning. The introduction of **process rewards** (rewarding intermediate steps rather than just outcomes) is particularly innovative. In robotics, getting close to the goal (trajectory alignment) often matters as much as reaching it, and standard RL struggles to reward "good effort" without dense environment signals. MDPR solves this elegantly by calculating rewards offline based on known ground-truth trajectories.

**Critical Perspective:**
While the results are promising, the reliance on "Ground Truth" trajectories for computing rewards (QACR and CTAR) means the method assumes high-quality demonstration data is available. If the expert demonstrations are noisy or suboptimal, the rewards will reinforce poor behavior. Additionally, the assumption of "offline" reward calculation limits the ability of the agent to adapt to truly unknown dynamics during the fine-tuning phase. Future iterations could potentially combine these offline process rewards with sparse online feedback once the policy is deployed. Nevertheless, the separation of "reward design" from "environment interaction" is a valuable architectural insight that decouples the difficulty of training from the safety/cost of real-world trials.