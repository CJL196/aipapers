# 1. Bibliographic Information
## 1.1. Title
Latent Action Pretraining from Videos

The title clearly states the paper's core contribution: a method for **pretraining** robot action models using **videos** by learning **latent actions**, which implies that explicit, ground-truth action labels are not required during the pretraining phase.

## 1.2. Authors
Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo.

The authors are from several top-tier academic and industrial research institutions, including **KAIST**, **University of Washington**, **Microsoft Research**, **NVIDIA**, and the **Allen Institute for AI**. This collaboration between leading academic labs and major AI industry players suggests a high-impact research effort with significant computational resources and expertise in both large-scale models and robotics.

## 1.3. Journal/Conference
The paper was published as a preprint on **arXiv**. Given the publication date and the caliber of the research and authors, it is likely intended for submission to a top-tier machine learning or robotics conference such as NeurIPS (Conference on Neural Information Processing Systems), ICML (International Conference on Machine Learning), or CoRL (Conference on Robot Learning).

## 1.4. Publication Year
2024

## 1.5. Abstract
The abstract introduces a novel unsupervised pretraining method called **Latent Action Pretraining (LAPA)** for Vision-Language-Action (VLA) models. The core problem addressed is the heavy reliance of existing VLA models on ground-truth robot action labels, which are costly to collect via human teleoperation and thus limit the scale of training data. LAPA overcomes this by learning from internet-scale videos that lack such labels.

The method consists of three main stages:
1.  **Action Quantization:** A VQ-VAE-based model is trained to learn discrete "latent actions" that represent the change between video frames.
2.  **Latent Pretraining:** A VLA model is then pretrained to predict these discrete latent actions based on visual observations and natural language task descriptions.
3.  **Finetuning:** Finally, the pretrained VLA is finetuned on a small amount of robot data with real action labels to map the learned latent actions to executable robot commands.

    Experimental results show that LAPA significantly outperforms existing techniques for learning from videos without action labels. More impressively, it also surpasses the state-of-the-art VLA model (which was trained with action labels) on various real-world manipulation tasks. The paper also demonstrates successful knowledge transfer even when pretraining exclusively on human manipulation videos, highlighting its potential for leveraging massive web-scale datasets to build robotics foundation models.

## 1.6. Original Source Link
- **Official Source (arXiv):** [https://arxiv.org/abs/2410.11758](https://arxiv.org/abs/2410.11758)
- **PDF Link:** [https://arxiv.org/pdf/2410.11758v2](https://arxiv.org/pdf/2410.11758v2)
- **Publication Status:** This is a preprint and has not yet been peer-reviewed or officially published in a conference or journal.

# 2. Executive Summary
## 2.1. Background & Motivation
The development of generalist robot agents hinges on training powerful foundation models that can understand complex instructions and interact with the physical world. **Vision-Language-Action (VLA) models**, which integrate large language models with visual understanding and action generation, have emerged as a promising paradigm. However, their progress is fundamentally bottlenecked by data.

The core problem is that state-of-the-art VLAs require massive datasets of `(vision, language, action)` triplets. The `action` component—precise robot commands like end-effector movements or joint torques—is typically collected through painstaking and expensive human teleoperation. This dependency on labeled robot data severely limits the diversity and scale of pretraining datasets, hindering the development of truly generalist foundation models.

In stark contrast, the internet contains a virtually limitless supply of videos depicting humans interacting with the world. This data is rich with examples of physical manipulation and common sense. However, it presents two major challenges:
1.  **Lack of Action Labels:** These videos do not come with the structured, numerical action labels that robots need.
2.  **Embodiment Gap:** The morphology and dynamics of a human hand are vastly different from a robot gripper. This "embodiment gap" makes it difficult to directly transfer human motions to a robot.

    This paper's entry point is an innovative idea to sidestep these challenges. Instead of trying to infer precise, ground-truth robot actions from videos, the authors propose to learn an abstract, **latent action space** directly from visual changes. This approach, named **Latent Action Pretraining (LAPA)**, aims to build a generalist action model by first learning a universal "vocabulary" of physical interactions from unlabeled videos and then mapping this vocabulary to a specific robot's capabilities using a small amount of labeled data.

## 2.2. Main Contributions / Findings
The paper makes several key contributions to the field of robot learning:

1.  **A Novel Unsupervised Pretraining Framework (LAPA):** The primary contribution is the proposal of `Latent Action Pretraining`, a three-stage method that enables VLA models to be pretrained on large-scale video data **without any ground-truth action labels**. This significantly broadens the scope of usable data for training robotics models.

2.  **State-of-the-Art Performance:** LAPA not only outperforms other methods that learn from unlabeled videos but, remarkably, also surpasses `OpenVLA`, a leading VLA model that was pretrained on nearly a million trajectories *with* ground-truth action labels. Specifically, LAPA achieves a **6.22% higher success rate** on real-world tasks.

3.  **Superior Pretraining Efficiency:** The LAPA pretraining process is shown to be over **30 times more computationally efficient** than that of `OpenVLA`. This efficiency gain is attributed to a more suitable model backbone and a much smaller, more abstract action space, which simplifies the learning problem.

4.  **Demonstrated Transfer from Human Videos:** The paper shows that pretraining LAPA solely on human manipulation videos (from the `Something-Something v2` dataset) yields a robot policy that outperforms a VLA pretrained on a large-scale robot dataset (`BridgeV2`). This is a crucial finding, as it validates the potential of using web-scale human video data to build powerful robot foundation models, despite the significant embodiment gap.

5.  **Potential as a Neural World Model:** The authors qualitatively demonstrate that the components of LAPA can be combined to form a "neural simulator." The LAPA policy predicts a latent action, and the decoder from the quantization stage predicts the resulting future frame, enabling closed-loop rollouts and planning entirely within the model's neural networks.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
### 3.1.1. Vision-Language-Action (VLA) Models
A **Vision-Language-Action (VLA) model** is a type of artificial intelligence model designed to control a robot. It integrates three modalities:
- **Vision:** It processes visual input from cameras to perceive the environment (e.g., identify objects, understand their state).
- **Language:** It understands natural language instructions given by a human (e.g., "pick up the red block").
- **Action:** It generates low-level commands to control the robot's motors (e.g., move the arm to position (x, y, z), close the gripper).

  VLAs are typically built upon the architecture of a **Vision-Language Model (VLM)**, which is pretrained on vast internet datasets of images and text. This VLM is then "fine-tuned" on robot-specific data, teaching it to map its rich visual and semantic understanding to physical actions. This allows VLAs to generalize to novel instructions and objects not seen during robot-specific training.

### 3.1.2. Vector Quantized Variational Autoencoder (VQ-VAE)
A **Vector Quantized Variational Autoencoder (VQ-VAE)** is a type of generative model that learns to compress and reconstruct data, such as images. Its key innovation is that it learns a **discrete** latent representation, unlike standard VAEs which use a continuous one.

It consists of three main parts:
1.  **Encoder:** A neural network that takes an input (e.g., an image) and maps it to a continuous latent vector in a lower-dimensional space.
2.  **Codebook:** A fixed-size collection of "embedding vectors" (e.g., 512 different vectors). This is the "dictionary" of discrete representations. The continuous vector from the encoder is replaced by the **closest** vector from this codebook. This step is called **vector quantization**.
3.  **Decoder:** A neural network that takes the discrete vector from the codebook and tries to reconstruct the original input.

    The model is trained to minimize the reconstruction error. By forcing the model to use a discrete code from the codebook, VQ-VAE learns a structured, compressed representation of the data. In LAPA, this mechanism is cleverly repurposed to create a "vocabulary" of discrete action tokens from continuous visual changes.

    ![Figure 8: Model architecture of our Latent Action Quantization Model.](images/8.jpg)
    *Figure 8 from the paper illustrates the architecture of the Latent Action Quantization Model, which is based on VQ-VAE principles.*

### 3.1.3. Behavior Cloning (BC)
**Behavior Cloning (BC)** is a simple and widely used approach in imitation learning. The goal is to train an agent (a "policy") to mimic the behavior of an expert. This is framed as a standard supervised learning problem.
- **Data:** A dataset of expert demonstrations is collected, where each data point is a pair of `(observation, action)`. The observation is what the expert saw, and the action is what the expert did.
- **Training:** A model (e.g., a neural network) is trained to predict the expert's `action` given the `observation`.
- **Goal:** The trained policy should be able to replicate the expert's behavior when faced with similar observations.

  In LAPA, behavior cloning is performed during the latent pretraining stage, where the model learns to predict the "expert" latent action tokens generated by the quantization model.

## 3.2. Previous Works
The paper positions itself relative to several key lines of research:

- **VLA Models with Labeled Actions:** This is the dominant paradigm. Works like **`RT-2`** (Brohan et al., 2023) and **`OpenVLA`** (Kim et al., 2024) have shown that fine-tuning large VLMs on extensive robot datasets (like the `Open X-Embodiment` dataset) with ground-truth actions leads to highly capable and generalist policies. The main drawback, which LAPA addresses, is the reliance on this expensive labeled data.

- **Learning from Unlabeled Videos:** Several methods have tried to leverage action-less videos.
    - **`VPT` (Video PreTraining)** (Baker et al., 2022): `VPT` first trains an **Inverse Dynamics Model (IDM)** on a small amount of *labeled* data. An IDM learns to predict the action that occurred between two consecutive states (frames). This trained IDM is then used to generate "pseudo-action" labels for a large unlabeled video dataset. Finally, a policy is trained on this pseudo-labeled data. LAPA differs by not requiring any labeled data to generate its latent actions.
    - **`UnIPI`** (Du et al., 2023): This method uses a text-conditioned video diffusion model to generate a plan of future frames. An IDM is then used to infer the actions needed to follow this visual plan. `UnIPI` focuses on planning through video generation, whereas LAPA focuses on learning a direct policy that outputs latent actions.

- **Latent Action Models:** The idea of latent actions is not entirely new, but its application in this context is.
    - **`GENIE`** (Bruce et al., 2024): `GENIE` learns a latent action space from videos to create **generative interactive environments**. A user can provide an action, and the model generates the next frame of a simulated world. LAPA adopts a similar VQ-VAE-based model for action quantization but uses it for a different goal: pretraining a policy for real-world robotics, not generating a game world.
    - Other works (Lynch et al., 2020) have used latent spaces to model action *multimodality* (i.e., many different action sequences can achieve the same goal), but they typically learn this latent space from ground-truth actions, whereas LAPA learns it directly from pixels.

## 3.3. Technological Evolution
The field of robot learning from demonstrations has evolved significantly:
1.  **Early Imitation Learning:** Simple policies trained on small, in-domain datasets of `(state, action)` pairs. These models had poor generalization.
2.  **Pretrained Visual Representations:** Researchers began pretraining the vision part of the policy on large image datasets (like ImageNet) or video datasets, then fine-tuning the whole policy on robot data. This improved visual understanding. An example is **`R3M`** (Nair et al., 2022).
3.  **End-to-End VLA Models:** With the rise of LLMs, the community moved to large, monolithic VLA models (`RT-2`, `OpenVLA`) pretrained on web-scale vision-language data and then fine-tuned on large-scale robot demonstration data. This brought unprecedented generalization to language and vision.
4.  **Leveraging Unlabeled Video:** Recognizing the robot data bottleneck, recent works (`VPT`, `UnIPI`) have tried to incorporate unlabeled videos, but often by trying to synthesize or infer ground-truth-like actions.
5.  **LAPA's Abstraction:** LAPA represents the next step in this evolution. Instead of trying to force web video into the mold of ground-truth robot actions, it creates a new, universal action abstraction layer. This decouples the pretraining from the specific embodiment of any single robot, making it more scalable and general.

## 3.4. Differentiation Analysis
Compared to prior work, LAPA's core innovation lies in **how it defines and uses "action" during pretraining**.

- **vs. `OpenVLA`/`ActionVLA`:** These methods perform behavior cloning on **ground-truth robot actions** (e.g., 7-dimensional end-effector poses). This ties the pretraining directly to specific robot embodiments and requires expensive labeled data. LAPA performs behavior cloning on **abstract, discrete latent actions** learned from pixels, making it unsupervised and embodiment-agnostic during pretraining.

- **vs. `VPT`:** `VPT` uses an IDM to generate **pseudo-labels** that are still meant to approximate ground-truth actions. Its effectiveness is thus dependent on the quality of the initial labeled data used to train the IDM and the IDM's ability to generalize. LAPA's latent actions are not trying to be ground-truth actions; they are an emergent representation of "visual change," which may be a more robust signal, especially across different embodiments (human vs. robot).

- **vs. `UnIPI`:** `UnIPI` is a multi-stage, model-based approach that relies on generating video rollouts and then inferring actions. This can be slow and prone to compounding errors in the generated video. LAPA trains a direct, reactive policy that maps observation and language to a latent action, which is conceptually simpler and more efficient at inference time.

  In summary, LAPA introduces a paradigm shift: **don't try to guess the exact robot actions from a human video; instead, learn a universal vocabulary of physical interactions and then translate that vocabulary to the robot's specific dialect.**

# 4. Methodology
The LAPA framework is implemented through a sequential three-stage process. The core idea is to first learn a universal, discrete action vocabulary from unlabeled videos, then pretrain a policy to use this vocabulary, and finally adapt this policy to a specific robot.

The overall workflow is visualized in the paper's overview figure:

![Figure : Overview of Latent Action Pretraining. (1) Latent Action Quantization: We first learn discrete latent actions in a fully unsupervised manner using the VQ-VAE objective (Detail in Figure 8).() Latent Pretraining:The VLM is trained to predict latent actions, essentially performing behavior cloning. After pretraining, we finetune LAPA on a small set of action-labeled trajectories to map the latent space to the end effector delta action space.](images/2.jpg)
*Figure 2 from the paper provides a high-level overview of the LAPA process, showing the two pretraining stages (Latent Action Quantization and Latent Pretraining) followed by the finetuning stage.*

## 4.1. Principles
The fundamental principle of LAPA is to decouple the problem of learning *what to do* (semantics and high-level planning) from *how to do it* (low-level motor control).
1.  The first stage, **Latent Action Quantization**, learns a universal dictionary of primitive physical changes (e.g., "move left," "push forward," "rotate clockwise") directly from pixels, without any human-defined labels. This dictionary takes the form of a discrete codebook, turning continuous motion into a set of "action tokens."
2.  The second stage, **Latent Pretraining**, teaches a large VLA model *what to do*. It learns to select the appropriate action token from the dictionary to make progress on a task described by a language instruction. This step leverages the powerful reasoning and planning capabilities of VLMs.
3.  The final stage, **Action Finetuning**, teaches the model *how to do it* on a specific robot. It learns the mapping from the abstract action tokens to the robot's concrete, continuous motor commands. Since the model already knows what to do, this final adaptation requires very little data.

## 4.2. Core Methodology In-depth (Layer by Layer)
### 4.2.1. Stage 1: Latent Action Quantization
This stage aims to create a discrete vocabulary of actions in a fully unsupervised manner. A VQ-VAE-based model is trained for this purpose.

- **Model Architecture:** The model is an encoder-decoder architecture based on `C-ViViT`.
    - **Input:** Two frames from a video, the current frame $x_t$ and a future frame $x_{t+H}$, where $H$ is a fixed window size.
    - **Encoder:**
        1. Both frames $x_t$ and $x_{t+H}$ are passed through a vision transformer (spatial transformer) to get patch embeddings.
        2. These embeddings are then processed by a temporal transformer to capture the motion between the frames, producing continuous embeddings $e_1$ (for $x_t$) and $e_2$ (for $x_{t+H}$).
        3. The difference, $d_1 = e_2 - e_1$, represents the continuous "delta" or change between the frames.
    - **Vector Quantization:** This is the core step where the continuous change $d_1$ is discretized. The model maintains a codebook $C$ of embedding vectors $\{z_k\}$. The latent action $z_1$ is found by searching for the codebook vector closest to $d_1$. This is expressed by the formula:
      \$
        z_1 = \arg\min_{z_k} |d_1 - z_k|^2
        \$
        Here, $z_1$ is the discrete latent action token—the index of the closest vector in the codebook.
    - **Noise Substitution VQ (NSVQ):** To prevent the model from using only a few codes in the codebook (a problem known as codebook collapse), the authors use NSVQ. Instead of passing the clean quantized vector $z_1$ to the decoder, they use a slightly perturbed version $\hat{d}_1$:
      \$
        \hat{d}_1 = d_1 + \frac{\lVert d_1 - z_1 \rVert}{\lVert v \rVert} v
        \$
        In this formula, $v \sim \mathcal{N}(0, 1)$ is a random noise vector. This technique encourages the model to explore and utilize the entire codebook.
    - **Decoder:** The decoder's job is to reconstruct the future frame $x_{t+H}$ given the current frame $x_t$ and the latent action. The paper uses cross-attention for this, where the decoder attends to the latent action $\hat{d}_1$ using the visual features of the current frame $p_1$ as queries. A stop-gradient `sg` is applied to $p_1$ to prevent representation collapse. The reconstruction $\hat{x}_2$ is generated as:
      \$
        \hat{x}_2 = D(\mathrm{Attn}(sg[p_1], \hat{d}_1, \hat{d}_1))
        \$
        Here, $D$ is the decoder network and $\mathrm{Attn}$ is the cross-attention mechanism.
    - **Training Objective:** The model is trained by minimizing the L2 reconstruction loss between the predicted future frame and the actual future frame:
      \$
        L = \lVert x_2 - \hat{x}_2 \rVert_2^2
        \$
        where $x_2$ is the ground-truth future frame $x_{t+H}$. After training, the **encoder** of this model serves as a tool to label any video with latent action tokens, and the **decoder** can be used as a world model to predict future states.

### 4.2.2. Stage 2: Latent Pretraining
With the trained quantization model from Stage 1, we can now pretrain the main VLA policy.

1.  **Data Labeling:** The encoder from Stage 1 is used as an offline **inverse dynamics model (IDM)**. It processes the entire pretraining dataset (e.g., internet videos) and, for each pair of consecutive frames $(x_t, x_{t+1})$, it generates a discrete latent action token $z_t$. The dataset is now transformed from `(video, language)` to $(image_t, language, latent_action_t)$.

2.  **Behavior Cloning on Latent Actions:** A pretrained VLM (the paper uses `LWM-Chat-1M`) is used as the backbone for the LAPA policy.
    - A new, small MLP layer (the "latent action head") is added to the VLM. This head has an output size equal to the codebook vocabulary size $|C|$.
    - The model is trained to predict the latent action token $z_t$ given the current image observation $x_t$ and the language instruction for the video clip.
    - During this training, the vision encoder of the VLM is kept frozen, while the language model parameters are unfrozen and trained. This allows the model to learn the mapping from its rich, pretrained language/vision understanding to the newly defined latent action space.

      This stage does not require any ground-truth robot actions, allowing it to scale to any video dataset with accompanying text descriptions.

### 4.2.3. Stage 3: Action Finetuning
The model pretrained in Stage 2 understands the high-level plan but cannot execute it on a real robot because its output is abstract tokens. This final stage bridges that gap.

1.  **Model Adaptation:** The latent action head from the previous stage is discarded. It is replaced with a new "action head" designed to output real robot commands.
2.  **Action Discretization:** The continuous robot action space (e.g., 7D end-effector deltas) is discretized into bins. For each dimension, the bins are created such that they contain an equal number of data points from the finetuning dataset. The action prediction task becomes a classification problem for each dimension.
3.  **Finetuning:** The model is finetuned on a **small** dataset of real robot demonstrations that include `(image, language, ground-truth_action)` tuples.
    - As before, the vision encoder remains frozen, while the LLM and the new action head are trained.
    - This process teaches the model to map the internal representations that previously led to a latent action token to the corresponding real robot action bin.

      Because the model has already learned the general semantics of actions and planning from the massive unlabeled dataset, this final finetuning step is very data-efficient.

# 5. Experimental Setup
## 5.1. Datasets
The authors use a diverse set of datasets for pretraining and finetuning to test LAPA across various scenarios, including in-domain, cross-task, cross-environment, and cross-embodiment generalization.

The following is the data from Table 3 of the original paper, summarizing the experimental design:

<table>
<thead>
<tr>
<th rowspan="2">Environment</th>
<th rowspan="2">Category</th>
<th colspan="2">Pretraining</th>
<th colspan="2">Fine-tuning</th>
</tr>
<tr>
<th>Dataset</th>
<th># Trajs</th>
<th>Dataset</th>
<th># Trajs</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">LangTable</td>
<td>In-Domain</td>
<td>Sim (All 5 tasks)</td>
<td>181k</td>
<td>5 Tasks (MT, MI)</td>
<td>1k</td>
</tr>
<tr>
<td>Cross-Task</td>
<td>Sim (All 5 tasks)</td>
<td>181k</td>
<td>1 Task (MI)</td>
<td>7k</td>
</tr>
<tr>
<td>Cross-Env</td>
<td>Real (All 5 tasks)</td>
<td>442k</td>
<td>5 tasks (MT, MI)</td>
<td>1k</td>
</tr>
<tr>
<td rowspan="2">SIMPLER</td>
<td>In-Domain</td>
<td>Bridgev2</td>
<td>60k</td>
<td>4 Tasks (MT)</td>
<td>100</td>
</tr>
<tr>
<td>Cross-Emb</td>
<td>Something v2</td>
<td>200k</td>
<td>4 Tasks (MT)</td>
<td>100</td>
</tr>
<tr>
<td rowspan="3">Real-World</td>
<td>Cross-Emb</td>
<td>Bridgev2</td>
<td>60k</td>
<td>3 tasks (MI)</td>
<td>450</td>
</tr>
<tr>
<td>Multi-Emb</td>
<td>Open-X</td>
<td>970k</td>
<td>3 tasks (MI)</td>
<td>450</td>
</tr>
<tr>
<td>Cross-Emb</td>
<td>Something v2</td>
<td>200k</td>
<td>3 tasks (MI)</td>
<td>450</td>
</tr>
</tbody>
</table>

- **`Language Table`:** A 2D physics simulation where a robot arm pushes blocks. It's used to test in-domain learning, cross-task generalization, and real-to-sim transfer (pretraining on real-world `Language Table` data and finetuning in sim).
- **`SIMPLER`:** A set of simulated 3D tabletop manipulation tasks with a 7-DOF WidowX robot arm. It's used to test transfer from robot data (`BridgeV2`) and human data (`Something v2`).
- **`BridgeV2`:** A large-scale real-world dataset collected on a WidowX robot arm. It's used for cross-embodiment pretraining (pretrain on WidowX, finetune on Franka).
- **`Open X-Embodiment (Open-X)`:** A massive, multi-embodiment dataset aggregating data from 20+ different robots. It's used to test pretraining in a highly diverse, multi-embodiment setting.
- **`Something Something v2 (Sthv2)`:** A large-scale dataset of short videos of humans performing basic actions with everyday objects. This dataset contains **no robots or robot actions**, making it a true test of learning from human-only video.

  The paper provides visual examples of these environments.

  ![Figure 9: Experimental Setups. (a) shows an example from the $4 4 0 \\mathrm { k }$ real-world trajectories (top) and the 181k simulation trajectoris ottom)fromheLanguageTableBencmark. shows thedifferent evaluatin tasks we use with the SIMPLER environment. (c) shows the three different tasks that we perform in the real-world.](images/9.jpg)
  *Figure 9 from the paper shows: (a) Language Table sim and real environments, (b) the four tasks in the SIMPLER environment, and (c) the three real-world manipulation tasks.*

## 5.2. Evaluation Metrics
- **Success Rate (%)**
    1.  **Conceptual Definition:** This is the primary metric for task completion. It measures the percentage of evaluation trials in which the robot successfully achieves the goal defined by the task instruction. A higher success rate indicates a more capable and reliable policy.
    2.  **Mathematical Formula:**
        \$
        \text{Success Rate} = \frac{\text{Number of Successful Trials}}{\text{Total Number of Trials}} \times 100\%
        \$
    3.  **Symbol Explanation:**
        - `Number of Successful Trials`: The count of episodes where the task was completed correctly.
        - `Total Number of Trials`: The total number of evaluation episodes run.

- **Partial Success Score**
    1.  **Conceptual Definition:** For complex, multi-stage tasks, a binary success/fail metric can be too coarse. A partial success score provides a more granular measure of progress by awarding points for completing sub-goals. For example, in a "pick and place" task, the robot might get partial credit for reaching the object, more for grasping it, and full credit for placing it correctly. This helps distinguish a policy that fails early from one that almost succeeds.
    2.  **Mathematical Formula:** There is no single formula; it is task-specific. The paper defines them as follows:
        - **Knocking:** 0.5 for reaching the correct object, 1.0 for knocking it over.
        - **Covering:** 0.33 for picking up the towel, 0.66 for reaching the target object with the towel, 1.0 for fully covering the object.
        - **Pick & Place:** 0.25 for reaching, 0.5 for grasping, 0.75 for moving towards the destination, 1.0 for placing correctly.
    3.  **Symbol Explanation:** N/A.

## 5.3. Baselines
LAPA is compared against a comprehensive set of baselines to demonstrate its effectiveness:

- **`SCRATCH`:** A baseline where the base VLM (`LWM-Chat-1M`) is finetuned directly on the downstream task data without any robotics-specific pretraining. This measures the benefit of the pretraining stage itself.
- **`UnIPI`** (Du et al., 2023): A representative method for learning from unlabeled videos. It uses a video diffusion model for planning and an IDM for action extraction. This provides a direct comparison against another action-free pretraining approach.
- **`VPT`** (Baker et al., 2022): Another key method for learning from unlabeled videos. It uses an IDM trained on labeled data to generate pseudo-action labels for pretraining. This tests the quality of LAPA's latent actions against inferred pseudo-actions.
- **`ActionVLA`:** A strong baseline created by the authors, representing the "upper bound" of conventional pretraining. It uses the same VLM backbone as LAPA but is pretrained on the same datasets **using the ground-truth action labels**. This isolates the impact of using latent actions versus real actions.
- **`OpenVLA`** (Kim et al., 2024): The current state-of-the-art open-source VLA model. It was pretrained on the massive `Open-X` dataset with ground-truth actions. Comparing against `OpenVLA` benchmarks LAPA against the best existing system in the field.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### 6.1.1. Simulation Results (Language Table)
The experiments in the `Language Table` environment test LAPA's capabilities in various generalization settings.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="2">In-domain (1k)</th>
<th colspan="2">Cross-task (7k)</th>
<th colspan="2">Cross-env (1k)</th>
</tr>
<tr>
<th>Seen</th>
<th>Unseen</th>
<th>Seen</th>
<th>Unseen</th>
<th>Seen</th>
<th>Unseen</th>
</tr>
</thead>
<tbody>
<tr>
<td>SCRATCH</td>
<td>15.6±9.2</td>
<td>15.2±8.3</td>
<td>27.2±13.6</td>
<td>22.4±11.0</td>
<td>15.6±9.2</td>
<td>15.2±8.3</td>
</tr>
<tr>
<td>UnIPI</td>
<td>22.0±12.5</td>
<td>13.2±7.7</td>
<td>20.8±12.0</td>
<td>16.0±9.1</td>
<td>13.6±8.6</td>
<td>12.0±7.5</td>
</tr>
<tr>
<td>VPT</td>
<td>44.0±7.5</td>
<td>32.8±4.6</td>
<td>72.0±6.8</td>
<td>60.8±6.6</td>
<td>18.0±7.7</td>
<td>18.4±9.7</td>
</tr>
<tr>
<td>LAPA</td>
<td>62.0±8.7</td>
<td>49.6±9.5</td>
<td>73.2±6.8</td>
<td>54.8±9.1</td>
<td>33.6±12.7</td>
<td>29.6±12.0</td>
</tr>
<tr>
<td>ActionVLa</td>
<td>77.0±3.5</td>
<td>58.8±6.6</td>
<td>77.0±3.5</td>
<td>58.8±6.6</td>
<td>64.8±5.2</td>
<td>54.0±7.0</td>
</tr>
</tbody>
</table>

- **In-Domain:** When pretrained and finetuned on the same task distribution, LAPA (62.0%) significantly outperforms other action-free methods like `UnIPI` (22.0%) and `VPT` (44.0%). It also closes a large portion of the gap to `ActionVLA` (77.0%), which uses ground-truth actions, demonstrating the effectiveness of latent action pretraining.
- **Cross-Task:** When finetuned on only one task (`separate`) but evaluated on all five, LAPA (73.2%) shows strong generalization, performing on par with `VPT` (72.0%) and nearly matching `ActionVLA`. This indicates the skills learned during latent pretraining are broad and can be retained.
- **Cross-Environment:** This is a key test of robustness. When pretrained on real-world data and finetuned on simulation data, LAPA (33.6%) still shows a significant positive transfer over `SCRATCH` (15.6%). In contrast, `VPT` (18.0%) shows almost no benefit, suggesting its IDM-based pseudo-labels are not robust to environment shifts, whereas LAPA's latent actions are.

### 6.1.2. Real-World Manipulation Results
These experiments benchmark LAPA against the state-of-the-art in a challenging real-world setting with a Franka robot arm, testing generalization to unseen objects and instructions.

![Figure 3: Real-world Tabletop Manipulation Results. We evaluate on a total of 54 rollouts for each model encompassing unseen object combinations, unseen objects and unseen instructions. Average success rate $( \\% )$ $\\pm$ StdErr are shown (detailed results provided in Appendix G.3).](images/3.jpg)
*Figure 3 from the paper summarizes the average success rates on the real-world tabletop tasks. LAPA (Open-X) achieves the highest performance.*

The following are the results from Table 2 of the original paper, breaking down performance by generalization type:

<table>
<thead>
<tr>
<th></th>
<th>Seen Obj. Unseen Combo</th>
<th>Unseen Obj.</th>
<th>Seen Obj. Unseen Instr.</th>
<th>AVG</th>
</tr>
</thead>
<tbody>
<tr>
<td>SCRaTCH</td>
<td>18.0</td>
<td>20.3</td>
<td>25.4</td>
<td>21.2</td>
</tr>
<tr>
<td>ACTIONVLA (Bridge)</td>
<td>38.3</td>
<td>31.8</td>
<td>27.7</td>
<td>32.6</td>
</tr>
<tr>
<td>OPENVLA (Bridge)</td>
<td>35.6</td>
<td>34.6</td>
<td>22.1</td>
<td>30.8</td>
</tr>
<tr>
<td>LAPA (Bridge)</td>
<td>43.4</td>
<td>31.4</td>
<td>35.6</td>
<td>36.8</td>
</tr>
<tr>
<td>OPENVLA (Open-X)</td>
<td>46.2</td>
<td>42.1</td>
<td>43.4</td>
<td>43.9</td>
</tr>
<tr>
<td>LAPA (Open-X)</td>
<td><strong>57.8</strong></td>
<td><strong>43.9</strong></td>
<td><strong>48.5</strong></td>
<td><strong>50.1</strong></td>
</tr>
<tr>
<td>LAPA (Human Videos)</td>
<td>36.5</td>
<td>37.4</td>
<td>28.1</td>
<td>34.0</td>
</tr>
</tbody>
</table>

- **`LAPA` vs. `OpenVLA` (State-of-the-Art):** The most significant result is that `LAPA (Open-X)` achieves an average success rate of **50.1%**, outperforming `OpenVLA (Open-X)` (43.9%) by a margin of **+6.2%**. This is remarkable because `OpenVLA` was pretrained with ground-truth actions, whereas LAPA was not. LAPA shows superior generalization across all three axes: unseen object combinations, unseen objects, and unseen instructions.
- **Cross-Embodiment Transfer (`BridgeV2` Pretraining):** When pretrained on `BridgeV2` (WidowX robot), LAPA (36.8%) outperforms both `ActionVLA` (32.6%) and `OpenVLA` (30.8%) pretrained on the same data. The authors hypothesize that pretraining with ground-truth actions can cause overfitting to the source robot's specific action space, which hurts transfer to a new robot (Franka). LAPA's embodiment-agnostic latent actions avoid this problem, leading to better cross-embodiment performance.
- **Transfer from Human Videos:** `LAPA (Human Videos)` achieves a **34.0%** success rate. This is a groundbreaking result. It not only shows positive transfer from human-only videos but also outperforms `OpenVLA (Bridge)` (30.8%), a model pretrained on a large-scale *robot* dataset. This strongly supports the central hypothesis that web-scale human videos are a viable and powerful data source for pretraining robot foundation models. The authors note it excels particularly with unseen objects (37.4%), likely due to the greater object diversity in the human video dataset.

### 6.1.3. Pretraining Efficiency
LAPA is not only more effective but also vastly more efficient. Pretraining `LAPA (Open-X)` took **272 H100-hours**. In contrast, pretraining `OpenVLA` took **21,500 A100-hours**. Adjusting for GPU differences, LAPA is roughly **30-40 times more efficient**. This is attributed to two factors:
1.  **Backbone Model:** The `LWM` backbone used by LAPA was already pretrained with a next-frame prediction objective, which likely gave it a strong prior for understanding motion.
2.  **Action Space Size:** LAPA predicts a small, discrete latent action (e.g., a vocabulary of $8^4 = 4096$ tokens), which is a much simpler classification problem than predicting 7 continuous action dimensions, each discretized into 256 bins (a space of $256^7$), as `OpenVLA` does.

## 6.2. Ablation Studies / Parameter Analysis
### 6.2.1. Scaling Laws
The authors investigate if LAPA benefits from scaling model size, data size, and latent action space size.

![Figure 5: Scaling Ablation Results of LAPA. We scale 4 dimensions of LAPA: model parameters (in millions), data size (ratio among Bridgev2), and the latentaction sequence and vocabulary size, and show the downstrem average success rate $( \\% )$ on the SIMPLER fine-tuning tasks.](images/5.jpg)
*Figure 5 from the paper shows that increasing model parameters, pretraining data size, and latent action space complexity all lead to improved downstream performance on the SIMPLER benchmark.*

The results confirm that LAPA follows expected scaling laws: performance consistently improves as the model gets bigger, is trained on more data, and uses a more expressive latent action space (larger vocabulary and sequence length). This suggests that LAPA's performance could be further improved by scaling up pretraining to even larger web-scale video datasets.

### 6.2.2. Latent Action Analysis
The authors perform qualitative analysis to understand what the learned latent actions represent.

- **Semantic Meaning:** By feeding a latent action to the decoder, they can visualize the motion it corresponds to. Figure 6 shows that the same latent action code (e.g., `[1,1,3,2]`) produces a similar semantic motion ("move down and left") across different robots and environments. This supports the claim that LAPA learns a shared, embodiment-agnostic action representation.

  ![Figure 6: Latent Action Analysis. We condition the current observation `x _ { 1 }` and quantized latent action to the decoder of the latent action quantization model.We observe that each latent action can be mapped into a semantic action. For example, latent action \[1,1,3,2\] corresponds to going down and left while \[3,2,0,1\] corresponds to going up a little bit.](images/6.jpg)
  *Figure 6 from the paper demonstrates that a single latent action code generates semantically similar movements across different embodiments, highlighting the universal nature of the learned action space.*

- **Closed-Loop Rollout as a World Model:** By feeding the LAPA policy's predicted latent action back into the decoder, the authors can generate a "neural simulation" of the task. Figure 7 shows LAPA successfully planning and visualizing the steps to "take the broccoli out of the pot," demonstrating its potential as a unified policy and world model.

  ![Figure 7: Closed loop rollout of LAPA. LAPA is conditioned on current image `x _ { 1 }` and language instruction of take the broccoli out of the pot We generate rollout images by conditining the decoder of Latent Action Quantization Model with latent actions generated by LAPA.](images/7.jpg)
  *Figure 7 from the paper shows a closed-loop rollout where LAPA predicts latent actions and the decoder visualizes the predicted future frames, effectively simulating the task completion.*

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces **Latent Action Pretraining (LAPA)**, a scalable and unsupervised pretraining method for Vision-Language-Action (VLA) models that completely removes the need for ground-truth action labels during the pretraining phase. By learning a discrete, universal latent action space from raw video pixels, LAPA effectively leverages large-scale, unlabeled video data, including human-centric videos from the web.

The key findings are compelling:
1.  LAPA significantly outperforms existing action-free pretraining methods across a range of simulation and real-world benchmarks.
2.  LAPA pretrained on a multi-embodiment dataset without action labels surpasses the performance of the state-of-the-art `OpenVLA` model, which was trained with nearly a million action-labeled trajectories.
3.  LAPA is over 30x more computationally efficient to pretrain than `OpenVLA`.
4.  LAPA demonstrates successful and positive transfer from human-only videos, opening a promising path toward building robotics foundation models using vast, readily available internet data.

## 7.2. Limitations & Future Work
The authors transparently acknowledge several limitations and areas for future research:

1.  **Fine-Grained Motion:** LAPA currently underperforms action-based pretraining on tasks requiring very precise, fine-grained control, such as grasping. This may be due to the coarse nature of the learned latent actions. Future work could explore increasing the latent action space's granularity.
2.  **Inference Latency:** Like other large VLA models, LAPA faces challenges with real-time inference speed. The authors suggest a hierarchical architecture (a large model for planning, a small model for fast execution) as a potential solution.
3.  **Beyond Manipulation:** The current work focuses on tabletop manipulation. The applicability of LAPA to other robotics domains like navigation, locomotion, or autonomous driving has not yet been explored, though the model's ability to learn camera movements suggests potential.

## 7.3. Personal Insights & Critique
This paper presents a significant and highly practical contribution to the field of robot learning.

**Strengths and Insights:**
- **Elegant Decoupling:** The core idea of decoupling the high-level "what" (semantic planning) from the low-level "how" (motor execution) by introducing a learned, abstract action layer is brilliant. It elegantly sidesteps the persistent embodiment gap problem that has plagued efforts to learn from human video.
- **Pragmatism and Scalability:** LAPA provides a clear, scalable, and resource-efficient path forward for the field. By removing the dependency on expensive teleoperated data for pretraining, it democratizes the creation of powerful robotics foundation models and aligns robot learning with the data-rich paradigms of NLP and computer vision.
- **Emergent World Model:** The demonstration of LAPA as a neural simulator is particularly exciting. This hints at a future where policies can perform internal "mental simulations" to plan, evaluate outcomes, and recover from errors, paving the way for more robust and intelligent model-based agents.

**Potential Issues and Critique:**
- **Sensitivity to Finetuning Data:** While the method reduces the *quantity* of labeled data needed, the final performance is still anchored by the quality and coverage of the small finetuning dataset. The process of mapping latent actions to real actions could be brittle if the finetuning data is sparse or noisy.
- **Implicit Bias of Quantization:** The latent actions are learned based on visual change. This may bias the model towards tasks with clear visual feedback and pose challenges for tasks where the critical action is subtle or involves non-visual modalities (e.g., force sensing). For example, distinguishing between "touching lightly" and "pressing hard" might be impossible from pixels alone.
- **Granularity vs. Expressiveness Trade-off:** The choice of the window size $H$ in the quantization stage seems critical. A small $H$ might capture fine-grained motion but miss long-term intent, while a large $H$ might capture high-level goals but gloss over crucial intermediate steps. A more dynamic or hierarchical approach to defining "action primitives" could be a valuable extension.

  Overall, `Latent Action Pretraining` is a landmark paper that offers a compelling solution to one of the most significant bottlenecks in robotics. Its success suggests that the future of generalist robots may indeed be built upon the foundation of simply watching the world.