# 1. Bibliographic Information

## 1.1. Title
SKILL-IL: Disentangling Skill and Knowledge in Multitask Imitation Learning

## 1.2. Authors
Xihan Bian, Oscar Mendez, and Simon Hadfield. The authors are affiliated with the University of Surrey, specifically within the Centre for Vision, Speech and Signal Processing (CVSSP). Their research focus typically involves robotics, computer vision, and machine learning.

## 1.3. Journal/Conference
This paper was published as a preprint on ArXiv in May 2022. Based on the formatting and content, it was presented at a major robotics conference (likely ICRA or IROS, though the document is a version of the work). CVSSP is a globally recognized research center in computer vision and robotics.

## 1.4. Publication Year
2022 (Specifically 2022-05-06).

## 1.5. Abstract
The paper proposes a novel framework called **SKILL-IL** for multi-task imitation learning. It hypothesizes that a robot's learned memory can be split into two distinct parts: **Skill** (the general ability to perform a task, like "how to drive") and **Knowledge** (specific information about the environment, like "the map of the city"). By disentangling these using a gated Variational Autoencoder (VAE) architecture, the authors demonstrate that an agent can generalize to unseen combinations of tasks and environments more effectively. The method outperformed existing state-of-the-art models by 30% in success rates across simulated environments and was validated on a real-world robot.

## 1.6. Original Source Link
*   **PDF Link:** [https://arxiv.org/pdf/2205.03130v2.pdf](https://arxiv.org/pdf/2205.03130v2.pdf)
*   **ArXiv Page:** [https://arxiv.org/abs/2205.03130](https://arxiv.org/abs/2205.03130)

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem addressed is the lack of **generalization** in multi-task **Imitation Learning (IL)**. In standard IL, an agent learns to mimic an expert. However, if an agent learns to "pick up a hammer in a kitchen," it often struggles to "pick up a hammer in a workshop" because it treats the environment and the task as a single, inseparable concept.

Humans do not learn this way. We possess **Procedural Memory** (skills) and **Declarative Memory** (knowledge). If you can cycle to the store, you can also cycle to work (transferring the skill of cycling to a new location) or drive to the store (transferring the knowledge of the route to a new vehicle). The authors aim to replicate this efficiency in robots, allowing them to re-combine learned skills and environmental knowledge for tasks they have never seen before (zero-shot learning).

## 2.2. Main Contributions / Findings
1.  **SKILL-IL Architecture:** A self-supervised framework using a **Gated Variational Autoencoder (VAE)** to partition the latent space into non-overlapping "Skill" and "Knowledge" subdomains.
2.  **Weakly Supervised Disentanglement:** A training strategy that uses pairs of experiences sharing either the same task or the same environment to "force" the network to store information in the correct partition.
3.  **Dynamic Loss Weighting:** An adaptive loss function that stabilizes training by prioritizing either reconstruction or policy accuracy depending on the current training mode.
4.  **Performance Leap:** The method achieved a 30% improvement in success rates over the prior state-of-the-art (SOTA) in complex, multi-step crafting tasks and showed significant efficiency gains in navigation.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, several core machine learning concepts are required:

*   **Imitation Learning (IL):** A branch of AI where an agent learns a policy $\pi$ by observing demonstrations from an expert. Instead of receiving a reward signal (as in Reinforcement Learning), the agent tries to minimize the difference between its actions and the expert's actions.
*   **Latent Space:** A compressed, hidden representation of data. When a neural network "looks" at an image, it transforms the millions of pixels into a smaller vector of numbers (the latent vector). This vector ideally captures the "essence" of the data.
*   **Variational Autoencoder (VAE):** A type of generative model that consists of an **Encoder** (turns data into a distribution in latent space) and a **Decoder** (reconstructs the data from the latent space). It uses a **Reconstruction Loss** to ensure the latent space holds enough information to recreate the input.
*   **Disentanglement:** The process of ensuring that different dimensions of the latent space represent different physical factors (e.g., one dimension represents "color," another "shape").

## 3.2. Previous Works
The paper builds heavily on **Compositional Plan Vectors (CPV)** proposed by Devin et al. [7]. 
*   **CPV Core Idea:** In multi-tasking, a task can be seen as a vector. If task $A$ is "pick up bread" and task $B$ is "eat bread," the sequence $A \rightarrow B$ should be represented by the vector sum $V_A + V_B$. 
*   **The CPV Formula:** $V_{0:T} = V_{0:t} + V_{t:T}$. This means the representation of the whole task (`0` to $T$) is the sum of the work done so far (`0` to $t$) and the work remaining ($t$ to $T$).

## 3.3. Technological Evolution & Differentiation
Traditionally, multi-task IL attempted to learn one massive policy for everything, which is data-inefficient. CPVs improved this by making tasks "additive." However, CPVs still "entangled" the environment with the action. **SKILL-IL** takes this a step further by splitting the CPV itself into two parts: one that cares about *what* is being done (Skill) and one that cares about *where* it is being done (Knowledge).

---

# 4. Methodology

## 4.1. Principles
The core intuition is **Partitioning**. By using a **Gated VAE**, the authors physically reserve certain "slots" in the latent vector for skill and others for knowledge. They then "blind" the gradients during training: if the agent is shown two different environments doing the same task, only the "Skill" slots are allowed to update.

The following figure (Figure 2 from the original paper) shows the system architecture:

![该图像是示意图，展示了多任务模仿学习中技能与知识的解缠方案。图中包括合成编码器、解缠解码器及策略网络，通过对环境上下文与任务技能的分离，提升了训练效率与任务成功率。](images/2.jpg)
*该图像是示意图，展示了多任务模仿学习中技能与知识的解缠方案。图中包括合成编码器、解缠解码器及策略网络，通过对环境上下文与任务技能的分离，提升了训练效率与任务成功率。*

## 4.2. Core Methodology In-depth

### 4.2.1. Compositional Task Embedding
The agent represents a task as an embedding $\vec{v}$. To handle multi-step tasks without forcing a strict order, the authors use commutativity ($\mathrm{A} + \mathrm{B} = \mathrm{B} + \mathrm{A}$). The "to-do" list for the agent at any time $t$ is calculated as the vector subtraction:
$$ \text{To-do Embedding} = \vec{v} - \vec{u} $$
Where $\vec{v}$ is the total task (start-to-end) and $\vec{u}$ is the progress (start-to-current).

The policy $\pi$ then decides the action $a_t$ based on the current visual observation $O_t$ and this to-do vector:
$$ a_t = \pi(a_t | O_t, \vec{v} - \vec{u}) $$

### 4.2.2. Gated Disentanglement
The latent vector $\vec{u}$ is split into two subdomains: $\vec{u} = [\vec{u}^s, \vec{u}^k]$, where $s$ is skill and $k$ is knowledge. The authors use a specific operator $\lfloor \rfloor$ which represents **gradient masking** (stopping the flow of learning to specific neurons).

The gated latent space $\Vec{u}$ is defined based on the training pair $(O, \hat{O})$:
$$
\Vec{u} = \left\{ \begin{array} { l l } { \left[ \Vec { u } ^ { s } , \lfloor \Vec { u } ^ { k } \rfloor \right] } & { \text{if } ( O , \hat { O } ) \in \mathcal { S } \text{ (Same Skill, Diff Env)} } \\ { \left[ \lfloor \Vec { u } ^ { s } \rfloor , \vec { u } ^ { k } \right] } & { \text{if } ( O , \hat { O } ) \in \mathcal { K } \text{ (Diff Skill, Same Env)} } \\ { \left[ \Vec { u } ^ { s } , \vec { u } ^ { k } \right] } & { \text{if } ( O , \hat { O } ) \in \mathcal { S } \cap \mathcal { K } \text{ (Original Pair)} \end{array} \right.
$$

### 4.2.3. Loss Functions
The model is trained using a combination of four distinct losses:

1.  **Reconstruction Loss ($L_{\delta}$):** Ensures the latent space can recreate the images.
    $$ L_{\delta}(O_{0:T}^{ref}, O_{0:t}, \hat{O}^{ref}, \hat{O}) = l_{\delta}(O_{0:T}^{ref}, \hat{O}^{ref}) + l_{\delta}(O_{0:t}, \hat{O}) $$
    where $l_{\delta}$ is the pixel-wise difference between the original observation $O$ and the reconstructed image $\hat{O}$.

2.  **Policy Loss ($L_a$):** The imitation learning objective.
    $$ L_a(O_t, \phi) = -\log(\pi(\hat{a}_t | O_t, g_{\phi}(O_{0:T}^{ref}) - g_{\phi}(O_{0:t}))) $$
    Here, $\hat{a}_t$ is the expert's action, and $g_{\phi}$ is the encoder function. This measures how "surprised" the agent is by the expert's choice.

3.  **Compositionality Loss ($L_C$):** Enforces the vector addition rule ($V_{0:t} + V_{t:T} = V_{0:T}$).
    $$ L_C(O_0, O_t, O_T) = l_m(g(O_{0:t}) + g(O_{t:T}^{ref}) - g(O_{0:T}^{ref})) $$
    where $l_m$ is a **Triplet Margin Loss** (a distance-based metric that ensures the sum stays close to the target).

4.  **Progress Loss ($L_P$):** Ensures the agent's trajectory embedding matches the expert's reference trajectory embedding.
    $$ L_P = l_m(g(O_{0:T}) - g(O_{0:T}^{ref})) $$

### 4.2.4. Dynamic Loss Weighting
To improve stability, the authors introduce $L_G$, which weights the policy and reconstruction losses differently based on whether the agent is currently learning a skill or knowledge:
$$ L_G = \begin{cases} \alpha L_a + \beta L_{\delta} & \text{if Skill Mode} \\ \epsilon \alpha L_a + \beta L_{\delta} & \text{if Knowledge Mode} \end{cases} $$
In **Skill Mode**, the agent focuses on the action ($\alpha$ is high). In **Knowledge Mode**, it focuses on environment details ($\beta$ is high).

---

# 5. Experimental Setup

## 5.1. Datasets
The authors used two primary environments:

1.  **Craftworld:** A 2D Minecraft-like world.
    *   **Tasks:** Chop trees, break rocks, make bread.
    *   **Complexity:** Sequences of up to 16 tasks.
    *   **Data Example:** An image showing a grid with a character, a tree, and a hammer. The "skill" is the sequence of moves to use the hammer; the "knowledge" is where the tree is located.
2.  **Learned Navigation:** 2D maps generated from real-world `gmapping` data.
    *   **Task:** Navigate from point A to point B.
    *   **Input:** Map images and camera views.

        The following figure (Figure 3) illustrates the training modes and data samples:

        ![Fig. 3. The different inputs for different training modes. In skill mode, the environment differs from the original but the agent is expected to perform the same task. In knowledge mode, the environment is the same but the agent is expected to perform a different task.](images/3.jpg)
        *该图像是示意图，展示了三种不同的训练模式：左侧为原始环境，任务为“吃面包”；中间为知识训练模式，任务为“砍树”；右侧为技能训练模式，任务为“吃面包”。这些模式分别展示了在不同环境和任务下的训练输入。*

## 5.2. Evaluation Metrics

1.  **Task Success Rate:**
    *   **Definition:** The percentage of episodes where the agent successfully reaches the final goal state.
    *   **Formula:** $\frac{\text{Successful Episodes}}{\text{Total Episodes}} \times 100\%$.
2.  **Average Episode Length:**
    *   **Definition:** The number of steps taken to complete the task. Lower is better (indicates efficiency).
    *   **Formula:** $\frac{1}{N} \sum_{i=1}^N T_i$ where $T_i$ is the number of steps in episode $i$.
3.  **Imitation Accuracy:**
    *   **Definition:** How often the agent's predicted action matches the expert's action at any given frame.

## 5.3. Baselines
*   **CPV-FULL [7]:** The previous state-of-the-art that uses compositional vectors but does not disentangle skill from knowledge.
*   **CPV-NAIVE:** A simpler version of CPV without the full architectural optimizations.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
SKILL-IL significantly outperformed the SOTA. In the most complex scenarios (16 skills in a sequence), the success rate was relatively low for both, but SKILL-IL was much more efficient. In the "4,4" task (4 tasks from 4 trajectories), SKILL-IL achieved a **26.3% success rate** compared to SOTA's **20.0%**, while being **twice as fast** (198 steps vs 379 steps).

## 6.2. Data Presentation (Tables)
The following are the results from Table I (Ablation Study) of the original paper:

| Model | Imitation accuracy | Success | Ep. length |
| :--- | :--- | :--- | :--- |
| CPV-FULL [7] | 66.42% | 65% | 69.31 |
| SKILL-no $O_t$ | 64.18% | 65% | 26.95 |
| SKILL | 70.61% | 84% | 19.77 |
| SKILL+FS | 70.89% | 89% | 19.52 |
| **SKILL+FS+DL** | 70.62% | **94%** | **17.88** |

*Note: FS = Fixed Sampling, DL = Dynamic Loss.*

The following are the results from Table III (Comparison against SOTA) of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">MODEL</th>
<th colspan="2">4 SKILLS</th>
<th colspan="2">8 SKILLS</th>
<th colspan="2">16 SKILLS</th>
<th colspan="2">1,1</th>
<th colspan="2">2,2</th>
<th colspan="2">4,4</th>
</tr>
<tr>
<th>Success</th>
<th>Length</th>
<th>Success</th>
<th>Length</th>
<th>Success</th>
<th>Length</th>
<th>Success</th>
<th>Length</th>
<th>Success</th>
<th>Length</th>
<th>Success</th>
<th>Length</th>
</tr>
</thead>
<tbody>
<tr>
<td>CPV-NAIVE</td>
<td>52.5</td>
<td>82.3</td>
<td>29.4</td>
<td>157.9</td>
<td>17.5</td>
<td>328.9</td>
<td>57.7</td>
<td>36.0</td>
<td>0.0</td>
<td>-</td>
<td>0.0</td>
<td>-</td>
</tr>
<tr>
<td>CPV-FULL</td>
<td>71.8</td>
<td>83.3</td>
<td>37.3</td>
<td>142.8</td>
<td>22.0</td>
<td>295.8</td>
<td>73.0</td>
<td>69.3</td>
<td>58.0</td>
<td>270.2</td>
<td>20.0</td>
<td>379.8</td>
</tr>
<tr>
<td><strong>SKILL</strong></td>
<td>61.3</td>
<td>**63.3**</td>
<td>37.5</td>
<td>**132.7**</td>
<td>20.0</td>
<td>**277.8**</td>
<td>**80.0**</td>
<td>**53.3**</td>
<td>55.0</td>
<td>**103.1**</td>
<td>**26.3**</td>
<td>**198.1**</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies
The authors found that **disentanglement is real**. As shown in Figure 5, the "Knowledge" latent can reconstruct the environment perfectly, but the "Skill" latent cannot. This proves the information has been successfully separated. 

![Fig. 5. The reconstructed image from the knowledge latent recreated the original image almost perfectly, full latent recreated the image without items unrelated to the current task (red hammer and purple house are not related to chop trees), and skill latent fails to generate an image that resembles the ground truth.](images/5.jpg)
*该图像是图表，展示了从知识潜在表示、技能潜在表示和完整潜在表示生成的图像。上方是原始图像（Ground Truth），下方是生成的图像。知识潜在几乎完美重建了原图，而技能潜在生成的图像与真实图像相差甚远。*

They also found a trade-off: allocating more space to "Knowledge" (KL) increased the success rate (98%), but allocating more to "Skill" (SL) made the agent faster (episode length 11.47), even if it failed more often.

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that partitioning a robot's memory into "Skill" and "Knowledge" subdomains significantly improves its ability to generalize in multi-task imitation learning. By using a gated VAE and a weak supervision training signal, the SKILL-IL framework creates a latent space that is not only more efficient but also more interpretable.

## 7.2. Limitations & Future Work
*   **Hard Gating:** The latent space is split into fixed, non-overlapping regions. In reality, some information might be both skill and knowledge.
*   **Simple Environments:** While real-robot tests were done, the primary environments (Craftworld and 2D Navigation) are relatively low-dimensional compared to complex 3D manipulation tasks.
*   **Future Direction:** Exploring "human-interpretable" latent spaces where we can manually adjust a robot's knowledge (e.g., "swap the map") without retraining.

## 7.3. Personal Insights & Critique
This paper provides an elegant solution to the **"Clever Hans" effect** in robotics—where a robot appears to learn a task but is actually just memorizing background pixels. By forcing the network to separate the "how" from the "where," the authors ensure the agent learns the actual mechanics of the task. 

One critique is the reliance on "Expert Reference Trajectories" ($O^{ref}$). In many real-world scenarios, we don't have perfect expert paths for every environment. Future work should look at how this disentanglement could be achieved through self-exploration or Reinforcement Learning, reducing the dependency on high-quality human data. However, the 50% reduction in episode length for complex tasks is a massive achievement for robotic efficiency.