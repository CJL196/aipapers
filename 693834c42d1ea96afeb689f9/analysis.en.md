# 1. Bibliographic Information

## 1.1. Title
Plan, Posture and Go: Towards Open-World Text-to-Motion Generation

## 1.2. Authors
The authors are Jinpeng Liu, Wenxun Dai, Yiji Cheng, Yansong Tang (from Tsinghua University), and Chunyu Wang, Xin Tong (from Microsoft Research Asia). The authors have backgrounds in computer vision, graphics, and AI, with notable contributions in human-centric generation and understanding.

## 1.3. Journal/Conference
The paper was submitted to arXiv, a preprint server. The publication date of December 22, 2023, suggests it was likely submitted for consideration at a major computer vision or graphics conference in 2024, such as CVPR or SIGGRAPH. While arXiv preprints are not peer-reviewed, they are a standard way for researchers to disseminate work quickly.

## 1.4. Publication Year
2023

## 1.5. Abstract
The paper addresses the challenge of generating human motions from "open-world" text descriptions, i.e., prompts not seen during training. Conventional methods struggle with this as they are trained on limited datasets. While some approaches use the CLIP model to align text and motion, they often produce limited and unrealistic "in-place" motions. To solve this, the paper introduces **PRO-Motion**, a "divide-and-conquer" framework. It comprises three modules:
1.  **Motion Planner**: Uses a Large Language Model (LLM) to break down a complex text prompt into a sequence of simple, structured "scripts" that describe key body postures.
2.  **Posture-Diffuser**: A diffusion model that generates a static 3D posture from each script. The simplicity of the scripts makes this step robust and generalizable.
3.  **Go-Diffuser**: Another diffusion model that takes the sequence of generated postures and predicts the overall body translation and rotation, creating a realistic, non-static final motion.
    The authors show through experiments that their method surpasses existing approaches in generating diverse and realistic motions from complex prompts like "Experiencing a profound sense of joy".

## 1.6. Original Source Link
- **Original Source:** [https://arxiv.org/abs/2312.14828](https://arxiv.org/abs/2312.14828)
- **PDF Link:** [https://arxiv.org/pdf/2312.14828.pdf](https://arxiv.org/pdf/2312.14828.pdf)
- **Publication Status:** This is a preprint and has not yet been published in a peer-reviewed journal or conference at the time of its release.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem is the limited generalization ability of existing **text-to-motion generation** models. These models are typically trained on datasets with a fixed set of text descriptions and corresponding motion captures. Consequently, when given a text prompt that is conceptually different from the training data (an "open-world" prompt), they fail to produce accurate or plausible motions.

Recent attempts to overcome this limitation have involved using large pre-trained vision-language models like **CLIP** to create a shared space between text and motion. The idea is to leverage CLIP's understanding of natural language to guide motion generation. However, this approach faces two critical challenges:
1.  **Semantic Gap:** The text space learned by CLIP (from natural images and their captions) is not perfectly aligned with the nuances of human motion descriptions. This makes the alignment between text and motion ineffective.
2.  **Lack of Temporal Priors:** CLIP processes information statically and lacks an inherent understanding of time and causality. This makes it difficult for models using it to generate poses in the correct chronological order, often resulting in unrealistic, jerky, or "in-place" motions that lack natural forward movement.

    The paper's innovative entry point is a **divide-and-conquer strategy**. Instead of trying to solve the difficult problem of mapping complex, open-world text directly to a complex, dynamic motion sequence, they break it down into three simpler, more manageable sub-problems:
1.  **Plan:** Use an LLM to "translate" the complex user prompt into a sequence of simple, structured instructions for key poses.
2.  **Posture:** Generate one static pose at a time from each simple instruction.
3.  **Go:** Synthesize a fluid, dynamic motion with realistic global movement from the sequence of static key poses.

## 2.2. Main Contributions / Findings
The paper's primary contributions are:

1.  **A Novel Three-Stage Framework (PRO-Motion):** The authors propose a modular framework consisting of a `motion planner`, `posture-diffuser`, and `go-diffuser`. This architecture systematically decomposes the complex text-to-motion task, making it more tractable and robust.

2.  **LLM-Powered Motion Planning:** A key innovation is the use of an LLM as a `motion planner`. The LLM leverages its commonsense reasoning to decompose an abstract or complex command (e.g., "feel happy") into a concrete, ordered sequence of key posture descriptions. These descriptions follow a simple, structured template, bridging the gap between ambiguous natural language and the precise language of body mechanics.

3.  **Hierarchical Generation Process:** The framework separates the generation of static body postures from the generation of global motion dynamics (translation and rotation).
    *   The `posture-diffuser` focuses only on generating a correct body pose from a simple script, a task it can generalize well to.
    *   The `go-diffuser` then infers the global movement (the "Go" part) from the sequence of poses, enabling the generation of realistic motions that travel through space, unlike the in-place motions of previous methods.

4.  **State-of-the-Art Open-World Performance:** The paper demonstrates through quantitative and qualitative experiments that PRO-Motion significantly outperforms existing methods. It can generate high-quality, realistic, and semantically correct motions for complex and unseen prompts, pushing the boundaries of open-world motion synthesis.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Denoising Diffusion Probabilistic Models (DDPMs)
DDPMs, or diffusion models, are a class of powerful generative models. They learn to create data (like images or, in this case, motion data) by reversing a process of gradually adding noise.
*   **Forward Process (Diffusion):** This is a fixed process where you start with a real data sample (e.g., a motion frame $x_0$) and repeatedly add a small amount of Gaussian noise over many timesteps $T$. By the end, at step $T$, the data becomes indistinguishable from pure random noise. The paper defines this with the formula:
    \$
    q ( x _ { t } | x _ { t - 1 } ) = N ( x _ { t } ; \sqrt { 1 - \beta _ { t } } x _ { t - 1 } , \beta _ { t } I )
    \$
    Here, $x_t$ is the data at step $t$, $\beta_t$ is a small constant determining how much noise is added at that step, and $N$ denotes a normal (Gaussian) distribution.

*   **Reverse Process (Denoising):** This is the generative part. The model (a neural network) learns to reverse the diffusion. Starting with pure noise ($x_T$), it iteratively removes a small amount of noise at each step ($t = T, T-1, ..., 1$) until it arrives at a clean data sample ($x_0$). To generate data conditioned on some input (like text), the model is given this condition at each step to guide the denoising process towards a desired output.

### 3.1.2. Large Language Models (LLMs)
LLMs, like GPT-3.5 used in this paper, are massive neural networks trained on vast amounts of text data. They excel at understanding and generating human language. A key capability leveraged here is **in-context learning**, where the LLM can perform a new task simply by being shown a few examples ("shots") in its input prompt, without needing to be retrained. In this paper, the LLM is not used to generate the motion directly, but to act as a "planner" or "translator," converting a high-level goal into a series of low-level, structured steps.

### 3.1.3. CLIP (Contrastive Language-Image Pre-training)
CLIP is a model trained by OpenAI on millions of image-text pairs from the internet. It learns to create a shared "embedding space" where the vector representation of an image and the vector representation of its corresponding text description are close together. This allows it to measure the semantic similarity between any given image and any given text prompt. Some previous works tried to adapt this for motion by rendering 3D poses as images and using CLIP to score how well they match a text prompt. However, this paper argues that CLIP's knowledge is about static images and is not optimized for the nuances or temporal flow of human motion.

### 3.1.4. SMPL (Skinned Multi-Person Linear model)
SMPL is a standard 3D statistical model of the human body. It provides a realistic and controllable way to represent human pose and shape. A "pose" is defined by a set of joint rotations, and these rotations are applied to a base body template to create a mesh of a person in that pose. The motion data in this paper is represented using SMPL parameters, which include:
*   **Pose Body:** Rotations for 21 body joints.
*   **Root Orientation:** The global orientation of the entire body.
*   **Translation:** The global position of the body in 3D space.

## 3.2. Previous Works

### 3.2.1. Conventional Text-to-Motion Generation
These methods (e.g., `Action2Motion`, `MDM`) train generative models like VAEs, GANs, or Diffusion Models directly on datasets of paired motion captures and text annotations (e.g., HumanML3D).
*   **Strength:** They can produce high-quality motions for text prompts that are similar to what they saw during training.
*   **Weakness:** They have poor **generalization**. They cannot handle "out-of-distribution" or "open-world" prompts because they have only learned the specific vocabulary and motion patterns present in their limited training set.

### 3.2.2. Open-Vocabulary Motion Generation (CLIP-based)
To address the generalization issue, some methods try to leverage CLIP's broad language understanding.
*   **AvatarCLIP:** This method uses CLIP to find poses that match a text description. It then uses these poses to guide a search within the latent space of a pre-trained motion VAE (Variational Autoencoder) to find a full motion. Its main drawbacks are that the search is time-consuming and it can't generate poses that are not already in its database.
*   **OOHMG (Being comes from Not-Being):** This work also uses CLIP to generate candidate poses but suffers from CLIP's lack of temporal understanding. The generated sequence of key poses often lacks correct chronological order, leading to unnatural or incorrect movements (e.g., a person bending down and then suddenly straightening up in the middle of a "bend over" motion).

### 3.2.3. Keyframe-based Motion Generation
This area focuses on generating motion based on a few specified keyframes (key poses). A common task is **motion in-betweening**, where a model generates the intermediate frames between a given start and end pose. This paper's `go-diffuser` is related but distinct: instead of being given key poses with their full 3D position and orientation, it is only given the *local* body postures and must *infer* the global translation and rotation (the "Go" part) itself.

## 3.3. Technological Evolution
The field has evolved from closed-world to open-world approaches:
1.  **Early Stage (Paired Data):** Models were trained on specific text-motion pairs, leading to high fidelity but low generalization (e.g., `MDM`).
2.  **Intermediate Stage (CLIP-based):** Researchers tried to use large pre-trained models like CLIP to bridge the vocabulary gap, enabling some open-world capability. However, this introduced new problems related to the mismatch between CLIP's domain (images) and motion, particularly the lack of temporal logic (e.g., `AvatarCLIP`, `OOHMG`).
3.  **Current Stage (LLM as Planner):** This paper represents a new direction where LLMs are used not just for understanding, but for **planning and decomposition**. The LLM acts as an intelligent layer that breaks a complex problem into simple, solvable parts, effectively sidestepping the issues of direct text-to-motion mapping and the limitations of CLIP.

## 3.4. Differentiation Analysis
The core innovation of PRO-Motion compared to previous works is its **intermediate representation and modular architecture**:

*   **vs. Paired Data Methods (MDM):** PRO-Motion is designed for open-world prompts, whereas MDM is limited to its training vocabulary. PRO-Motion achieves this by using the LLM to translate any prompt into a structured format it can understand.
*   **vs. CLIP-based Methods (AvatarCLIP, OOHMG):**
    *   **Better Semantic Alignment:** Instead of relying on the noisy alignment between CLIP's image-text space and motion, PRO-Motion uses an LLM to generate posture descriptions using a simple, controlled vocabulary. This creates a much cleaner and more direct link between language and pose.
    *   **Correct Temporal Ordering:** The LLM's planning capability inherently produces a sequence of key poses in a logical, chronological order. This solves the major flaw of CLIP-based methods, which often produce jumbled pose sequences.
    *   **Realistic Global Motion:** The dedicated `go-diffuser` module explicitly models global translation and rotation, allowing the character to move naturally through space, overcoming the "in-place" motion limitation of prior work.

        The overall paradigm is shown clearly in Figure 2 from the paper.

        ![Figure 2. Comparison of different paradigms for text-to-motion generation. (a) Most existing models leverage the generative models \[22, 33, 41\] to construct the relationship between text and motion based on text-motion pairs. (b) Some methods render 3D poses to images and employ the image space of CLIP to align text with poses. Then they reconstruct the motion in the local dimension based on the poses. (c) Conversely, we decompose motion descriptions into structured pose descriptions. Then we generate poses based on corresponding pose descriptions. Finally, we reconstruct the motion in local and global dimensions. "Gen.", "Decomp.", "Desc.", "Rec." stand for "Generative model", "Decompose", "Pose Description" and "Reconstruction" respectively.](images/2.jpg)
        *该图像是图示 2，展示了三种文本到运动生成的不同范式。(a) 利用生成模型构建文本与运动的关系。(b) 生成姿势并重构室内运动。(c) 通过分解运动描述生成姿势，最终重建运动。*

# 4. Methodology

The paper proposes **PRO-Motion**, a framework that decomposes the text-to-motion task into three stages: **Plan**, **Posture**, and **Go**.

## 4.1. Principles
The core idea is **divide and conquer**. Generating a long, complex motion from an abstract sentence is extremely difficult. However, the task becomes much easier if it's broken down:
1.  A human-like planner (the LLM) first imagines the key moments of the motion and describes them simply.
2.  A posture generator creates a static 3D snapshot for each simple description.
3.  A motion synthesizer then animates these snapshots, filling in the gaps and adding realistic global movement.

    This hierarchical approach isolates complexities. The LLM handles the ambiguity of natural language, the `posture-diffuser` handles the geometry of the human body, and the `go-diffuser` handles the dynamics of motion.

The full architecture is depicted in the figure below (a combination of Figures 3 and 4 from the paper).

![该图像是示意图，展示了PRO-Motion框架的三个模块：运动规划器、姿势扩散器和动作扩散器。用户提示经过运动规划器生成的姿势描述，随后姿势扩散器和动作扩散器分别负责将描述转化为姿势并估计身体动态。图中使用箭头表示不同模块之间的关系，包含了平移、旋转等动作参数。](images/3.jpg)
*该图像是示意图，展示了PRO-Motion框架的三个模块：运动规划器、姿势扩散器和动作扩散器。用户提示经过运动规划器生成的姿势描述，随后姿势扩散器和动作扩散器分别负责将描述转化为姿势并估计身体动态。图中使用箭头表示不同模块之间的关系，包含了平移、旋转等动作参数。*

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Stage 1: Motion Planner (Plan)
This stage uses a Large Language Model (specifically, GPT-3.5) to convert a high-level user prompt into a sequence of low-level, structured posture descriptions or "scripts".

*   **Input:** A user's text prompt (e.g., "Jump on one foot") and a carefully designed system prompt.
*   **Process (In-context Learning):** The system prompt instructs the LLM on its task and provides constraints. It contains:
    1.  **Task Definition:** An overall goal, including the desired frames per second (FPS) and the total number of keyframes to generate.
    2.  **Five Fundamental Rules:** These rules force the LLM to generate descriptions in a structured, compositional way, using a limited vocabulary. This is critical for reducing the complexity for the next stage. The rules concern:
        *   **Bending:** Degree of bend for joints like elbows and knees (e.g., 'slightly bent').
        *   **Relative Distances:** Distances between body parts (e.g., hands are 'shoulder width apart').
        *   **Relative Positions:** Positional relationships (e.g., 'left foot is behind right foot').
        *   **Orientation:** Whether a limb is 'vertical' or 'horizontal'.
        *   **Ground Contact:** Which body parts are 'on the ground'.
    3.  **Output Format:** A strict JSON-like format for the sequence of descriptions (e.g., ${"F1": "description_1", "F2": "description_2", ...}$).
    4.  **Examples:** A few sample pose descriptions to guide the LLM's generation process.
*   **Output:** A sequence of $F$ scripts, where each script $t_i$ describes a single key posture in the intended motion. For example, for "Jump on one foot," the LLM might generate scripts for "crouching," "pushing off," "in the air," and "landing."

### 4.2.2. Stage 2: Posture-Diffuser and Posture Planning (Posture)
This stage takes the sequence of scripts from the planner and generates a corresponding sequence of high-quality, static 3D key poses. This is a two-step process.

#### 4.2.2.1. Posture-Diffuser
The `posture-diffuser` is a conditional diffusion model that generates a single 3D pose $x_0$ conditioned on a single text script $c$.

*   **Architecture:** As shown in Figure 4(a), the model is a stack of layers. Each layer contains:
    *   A **residual block** that processes the pose features and incorporates the diffusion timestep $t$.
    *   A **cross-modal transformer block** that fuses the text information. The current (noisy) pose feature acts as the `query`, while the text script embeddings (from a frozen `DistilBERT` model) act as the `key` and `value`. This is a standard cross-attention mechanism.
*   **Training:** The model is trained to predict the original clean pose $x_0$ from a noised version $x_t$. The objective function is a simple mean squared error loss:
    \$
    \operatorname* { m i n } _ { \theta } \mathcal { L } = \mathbb { E } _ { x _ { 0 } , t , c } \left[ | | x _ { 0 } - f _ { \theta } ( x _ { t } , t , c ) | | _ { 2 } ^ { 2 } \right]
    \$
    where $f_\theta$ is the diffusion model network, $x_0$ is the ground-truth pose, $c$ is the corresponding script, and $x_t$ is the pose after adding noise for $t$ steps.
*   **Classifier-Free Guidance:** To improve the adherence of the generated pose to the text condition, the model uses classifier-free guidance. During training, the text condition $c$ is randomly dropped (set to a null token $\emptyset$) with some probability. At inference time, the model makes two predictions—one with the condition and one without—and extrapolates in the direction of the condition. The final prediction $\hat{x}_0$ is calculated as:
    \$
    f _ { \theta } ^ { w } ( x _ { t } , t , c ) = f _ { \theta } ( x _ { t } , t , \mathcal { O } ) + w \cdot ( f _ { \theta } ( x _ { t } , t , c ) - f _ { \theta } ( x _ { t } , t , \mathcal { O } ) )
    \$
    Here, $w$ is the guidance scale, which controls the trade-off between diversity and fidelity to the text prompt.

    ![Figure 4. Illustration of our Dual-Diffusion model. (a) PostureDiffuser module is designed to predict the original pose conditioned by the pose description. The model consists of $N$ identical layers, with each layer featuring a residual block for incorporating time step information and a cross-modal transformer block for integrating the condition text. (b) `G o` -Diffuser module serves the function of obtaining motion with translation and rotation from discrete key poses without global information. In this module, the key poses obtained from Sec. 3.3 are regarded as independent tokens. We perform attention operations \[87\] between these tokens and noised motion independently, which can significantly improve the perception ability between every condition pose and motion sequence.](images/4.jpg)
    *该图像是示意图，展示了我们提出的Dual-Diffusion模型的Posture-Diffuser模块（a）和Go-Diffuser模块（b）。Posture-Diffuser模块包含$N$个层，每层包括残差块和跨模态变换块，用于将姿势描述转化为姿势；Go-Diffuser模块从离散的关键姿势中获取运动的平移和旋转，进一步生成逼真的运动序列。*

#### 4.2.2.2. Posture Planning
Because diffusion models are stochastic, running the `posture-diffuser` for the same script multiple times will produce different valid poses. The Posture Planning module selects the best sequence of poses from these candidates to ensure the final motion is smooth and logical.

*   **Process:** For each of the $F$ scripts $\{t_i\}_{i=1}^F$, the `posture-diffuser` generates $L$ candidate poses $\{p_j^i\}_{j=1}^L$. The goal is to find the best path $G = \{g_1, g_2, ..., g_F\}$ where $g_i$ is the index of the chosen pose for the $i$-th frame. This is framed as a search problem solved by the **Viterbi algorithm**.
*   **Objective:** The algorithm optimizes two criteria simultaneously:
    1.  **Temporal Coherence (Smoothness):** Adjacent poses in the sequence should be similar. This is captured by a **transition probability** $A^i_{jk}$, which measures the similarity between pose $j$ from frame `i-1` and pose $k$ from frame $i$.
        \$
        A _ { j k } ^ { i } = \frac { \exp { \left( \Theta { ( p _ { j } ^ { i - 1 } ) } ^ { T } \Theta ( p _ { k } ^ { i } ) \right) } } { \sum _ { l = 1 } ^ { L } \exp { \left( \Theta { ( p _ { j } ^ { i - 1 } ) } ^ { T } \Theta ( p _ { l } ^ { i } ) \right) } }
        \$
        Here, $\Theta$ is a pose encoder that maps a pose to a feature vector. The formula is a softmax over the dot products of the feature vectors of candidate poses.
    2.  **Semantic Alignment:** Each chosen pose should strongly match its corresponding text script. This is captured by an **emission probability** $E^i_j$, which measures the similarity between script $t_i$ and candidate pose $p_j^i$.
        \$
        E _ { j } ^ { i } = \frac { \exp \Big ( \Phi ( t _ { i } ) ^ { T } \Theta ( p _ { j } ^ { i } ) \Big ) } { \sum _ { l = 1 } ^ { L } \exp \Big ( \Phi ( t _ { i } ) ^ { T } \Theta ( p _ { l } ^ { i } ) \Big ) }
        \$
        Here, $\Phi$ is a text encoder. This is a softmax over the dot products of the text and pose feature vectors.

*   **Final Path Selection:** The Viterbi algorithm finds the pose path $G$ that maximizes the joint probability of the entire sequence:
    \$
    \underset { G } { \arg \operatorname* { m a x } } P ( G ) = E _ { g _ { 1 } } ^ { 1 } \prod _ { i = 2 } ^ { F } E _ { g _ { i } } ^ { i } A _ { g _ { i - 1 } g _ { i } } ^ { i }
    \$

### 4.2.3. Stage 3: Go-Diffuser (Go)
The final stage takes the sequence of static key poses (which only have local joint rotations) and generates a full, fluid motion sequence that includes realistic global translation and rotation.

*   **Goal:** To perform interpolation between key poses and, crucially, to predict the global movement of the character. For example, if the key poses show "left foot forward" then "right foot forward," the `go-diffuser` should infer that the character is walking forward.
*   **Architecture:** This is also a conditional diffusion model, but with a **Transformer encoder** architecture, which is well-suited for sequence modeling (see Figure 4(b)).
*   **Input:**
    1.  A noised full-length motion sequence $x_t^{1:N}$.
    2.  The diffusion timestep $t$.
    3.  The sequence of $F$ selected key poses $\{p_{g_i}^i\}_{i=1}^F$ as the condition.
*   **Conditioning Mechanism:** The key poses are treated as a set of discrete conditional tokens. They are projected into an embedding space and fed to the transformer encoder. The attention mechanism allows each frame of the motion sequence to attend to all the key poses, enabling the model to understand the relationship between the local postures and the required global dynamics.
*   **Output:** The denoised, full-length motion sequence $\hat{x}_0^{1:N}$, which now includes both local body poses and global root translation and orientation, resulting in a realistic animation.

# 5. Experimental Setup

## 5.1. Datasets
The authors use several publicly available datasets for training and evaluation:
*   **AMASS (Archive of Motion Capture as Surface Shapes):** A large-scale database that unifies many different motion capture datasets. It provides over 40 hours of 3D human motion data represented using the SMPL model, but without text descriptions. It's used for training the motion VAE in baselines and likely for pre-training motion priors.
*   **PoseScript:** A dataset containing static 3D human poses extracted from AMASS. Crucially, it provides detailed, human-written text descriptions for each pose (`PoseScript-H`) and automatically generated captions (`PoseScript-A`). `PoseScript-A` is used to train the `posture-diffuser`.
*   **HumanML3D:** A widely-used dataset for text-to-motion research. It provides natural language descriptions for thousands of motion clips from AMASS. It is used to train supervised baselines like `MDM`.
*   **Motion-X:** A large-scale dataset with expressive, whole-body motions and detailed language descriptions. The authors create two specific test sets from this to evaluate open-world generation:
    1.  **ood368 subset:** They took a subset of Motion-X called IDEA-400 and filtered out any text-motion pairs that were too similar to the HumanML3D training set (using a sentence transformer similarity score). This ensures the test set is truly "out-of-distribution" (ood). This resulted in 368 pairs.
    2.  **kungfu subset:** A specific subset from Motion-X focusing on kungfu movements, which are complex and not well-represented in standard datasets.

## 5.2. Evaluation Metrics
The paper uses several metrics to evaluate the quality of the generated motions.

### 5.2.1. R-Precision
*   **Conceptual Definition:** This metric measures how well the generated motion matches its corresponding ground-truth text prompt compared to other "distractor" prompts. For each generated motion, its feature vector is compared against the feature vectors of a pool of text descriptions (one correct, many incorrect). If the correct text is ranked in the top-K closest descriptions, it's a hit. `R@K` is the percentage of times the correct description is in the top K. Higher is better.
*   **Mathematical Formula:** Let $f_m$ be the feature vector for a generated motion and $\{f_{t_1}, f_{t_2}, ..., f_{t_N}\}$ be the feature vectors for a pool of text descriptions, where $t_1$ is the ground truth. R-Precision @ K is the accuracy of:
    \$
    \text{rank}( \text{dist}(f_m, f_{t_1}) ) \leq K
    \$
    where $\text{rank}$ is the rank of the ground-truth text's distance among all distances, and $\text{dist}$ is typically Euclidean distance.

### 5.2.2. Frechet Inception Distance (FID)
*   **Conceptual Definition:** FID measures the similarity between the distribution of generated motions and the distribution of real motions. A lower FID score means the two distributions are more similar, indicating that the generated motions are more realistic and diverse. It's widely used in image generation and adapted here for motion.
*   **Mathematical Formula:**
    \$
    \mathrm{FID}(x, g) = \left\| \mu_x - \mu_g \right\|_2^2 + \mathrm{Tr}\left( \Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2} \right)
    \$
*   **Symbol Explanation:**
    *   $\mu_x$ and $\mu_g$: The mean of the feature vectors of the real data ($x$) and generated data ($g$), respectively.
    *   $\Sigma_x$ and $\Sigma_g$: The covariance matrices of the feature vectors for the real and generated data.
    *   $\mathrm{Tr}$: The trace of a matrix (sum of the diagonal elements).
    *   Lower FID is better.

### 5.2.3. MultiModal Distance (MM-Dist)
*   **Conceptual Definition:** This is a straightforward metric that measures the average distance between the feature vector of a generated motion and the feature vector of its corresponding ground-truth text description. A smaller distance indicates better alignment between the generated motion and the input prompt.
*   **Mathematical Formula:**
    \$
    \text{MM-Dist} = \frac{1}{N} \sum_{i=1}^{N} \| f_{m_i} - f_{t_i} \|_2
    \$
*   **Symbol Explanation:**
    *   $N$: The total number of samples.
    *   $f_{m_i}$: The feature vector of the $i$-th generated motion.
    *   $f_{t_i}$: The feature vector of the corresponding $i$-th ground-truth text.
    *   Lower MM-Dist is better.

### 5.2.4. Average Positional Error (APE) and Average Variance Error (AVE)
These metrics are used specifically to evaluate the `go-diffuser`'s ability to reconstruct motion accurately.
*   **Average Position Error (APE):** The average L2 distance between the joint positions of the generated motion and the ground-truth motion, over all frames and samples. Lower is better.
    \$
    A P E [ j ] = \frac { 1 } { N F } \sum _ { n \in N } \sum _ { f \in F } \left\| \boldsymbol { H } _ { f } \left[ j \right] - \hat { \boldsymbol { H } } _ { f } \left[ j \right] \right\| _ { 2 }
    \$
    *   $j$: a specific joint.
    *   `N, F`: number of samples and frames.
    *   $\boldsymbol{H}_f[j]$ and $\hat{\boldsymbol{H}}_f[j]$: ground-truth and generated positions of joint $j$ at frame $f$.

*   **Average Variance Error (AVE):** Measures the difference in the variance of joint positions over time between the generated and ground-truth motions. This captures whether the *amount* of movement is realistic. Lower is better.
    \$
    A V E [ j ] = { \frac { 1 } { N } } \sum _ { n \in N } \left\| \delta \left[ j \right] - \hat { \delta } \left[ j \right] \right\| _ { 2 }
    \$
    *   $\delta[j]$ and $\hat{\delta}[j]$: the temporal variance of joint $j$ for the ground-truth and generated motion.

## 5.3. Baselines
The paper compares PRO-Motion against several representative methods:
*   **MDM (Human Motion Diffusion Model):** A state-of-the-art supervised diffusion model trained on HumanML3D. Represents the performance of methods confined to their training data.
*   **MotionCLIP:** A supervised method that also aims for open-vocabulary generation by exposing the motion generation process to CLIP space.
*   **Codebook+Interpolation:** A simplified baseline where poses are selected from a fixed codebook (VPoserCodebook) via similarity matching, and motion is generated by simple interpolation.
*   **AvatarCLIP:** An optimization-based method that uses CLIP to guide a search in a motion VAE's latent space.
*   **OOHMG:** Another open-vocabulary method that uses CLIP image features to generate poses. It represents a key competitor in open-world generation.

    For the `Go-Diffuser` ablation, two baselines were used:
*   **Regression:** A simple MLP-based network to predict global information.
*   **Baseline[62]:** A baseline inspired by prior work where pose sequence features are extracted and used as a single condition for a diffusion model.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main results for open-world text-to-motion generation are presented in Table 1, comparing PRO-Motion against the baselines on the two challenging, out-of-distribution test sets.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="4">Text-motion</th>
<th rowspan="2">FID ↓</th>
<th rowspan="2">MultiModal Dist ↓</th>
<th rowspan="2">Smooth →</th>
</tr>
<tr>
<th>R@10 ↑</th>
<th>R@20 ↑</th>
<th>R@30 ↑</th>
<th>MedR ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8">"test on ood368 subset"</td>
</tr>
<tr>
<td>MDM [85]</td>
<td>17.81</td>
<td>34.06</td>
<td>48.75</td>
<td>31.20</td>
<td>3.500541</td>
<td>2.613644</td>
<td>0.000114</td>
</tr>
<tr>
<td>MotionCLIP [84]</td>
<td>16.25</td>
<td>35.62</td>
<td>52.81</td>
<td>28.90</td>
<td>2.227522</td>
<td>2.288905</td>
<td>0.000073</td>
</tr>
<tr>
<td>Codebook+Interpolation [34]</td>
<td>15.62</td>
<td>31.25</td>
<td>46.56</td>
<td>32.80</td>
<td>4.084785</td>
<td>2.516041</td>
<td>0.000146</td>
</tr>
<tr>
<td>AvatarCLIP [34]</td>
<td>15.31</td>
<td>31.56</td>
<td>47.19</td>
<td>32.60</td>
<td>4.181952</td>
<td>2.449695</td>
<td>0.000146</td>
</tr>
<tr>
<td>OOHMG [44]</td>
<td>15.62</td>
<td>34.06</td>
<td>48.75</td>
<td>29.80</td>
<td>3.982753</td>
<td>2.149275</td>
<td>0.000758</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>20.25</b></td>
<td><b>36.56</b></td>
<td><b>53.14</b></td>
<td><b>26.10</b></td>
<td><b>1.488678</b></td>
<td><b>1.534521</b></td>
<td><b>0.001312</b></td>
</tr>
<tr>
<td colspan="8">"test on kungfu subset"</td>
</tr>
<tr>
<td>MDM [85]</td>
<td>12.50</td>
<td>29.69</td>
<td>42.19</td>
<td>37.50</td>
<td>12.060187</td>
<td>3.725436</td>
<td>0.000735</td>
</tr>
<tr>
<td>MotionCLIP [84]</td>
<td>15.62</td>
<td>29.69</td>
<td>46.88</td>
<td>32.50</td>
<td>17.414746</td>
<td>4.297871</td>
<td>0.000123</td>
</tr>
<tr>
<td>Codebook+Interpolation [34]</td>
<td>10.94</td>
<td>20.31</td>
<td>29.69</td>
<td>37.50</td>
<td>2.521690</td>
<td>2.764137</td>
<td>0.000138</td>
</tr>
<tr>
<td>AvatarCLIP [34]</td>
<td>15.62</td>
<td>31.25</td>
<td>46.88</td>
<td>32.50</td>
<td>1.966764</td>
<td>2.497678</td>
<td>0.000171</td>
</tr>
<tr>
<td>OOHMG [44]</td>
<td>14.06</td>
<td>32.81</td>
<td>48.44</td>
<td>32.50</td>
<td>4.904853</td>
<td>2.471666</td>
<td>0.000847</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>20.31</b></td>
<td><b>34.38</b></td>
<td><b>50.00</b></td>
<td><b>31.00</b></td>
<td><b>4.124218</b></td>
<td><b>2.374380</b></td>
<td><b>0.001559</b></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Semantic Alignment (R-Precision, MedR, MM-Dist):** On both datasets, PRO-Motion (`Ours`) consistently achieves the highest `R-Precision` scores and the lowest (best) `MedR` (Median Rank) and `MultiModal Distance`. This strongly indicates that the motions generated by PRO-Motion are semantically much better aligned with the text prompts than any of the baselines. The LLM-based planning stage is clearly effective at preserving the intent of the original prompt.
*   **Motion Quality and Realism (FID):** On the `ood368` subset, PRO-Motion achieves a drastically lower (better) `FID` of 1.48, compared to the next best (MotionCLIP at 2.22). This signifies that the distribution of motions generated by PRO-Motion is much closer to the distribution of real human motions. On the `kungfu` subset, while AvatarCLIP has a lower FID, the authors note that CLIP-based methods struggle with temporal order. PRO-Motion's FID is still very competitive and significantly better than most other baselines.
*   **Qualitative Analysis (Figure 5):** The visual results confirm the quantitative findings. For the prompt "bends over," CLIP-based methods like MotionCLIP and AvatarCLIP produce an incorrect sequence of poses (bending then unbending). For the complex, unseen prompt "bury one's head and cry, and finally crouched down," the supervised `MDM` fails completely, while PRO-Motion generates a coherent and emotionally appropriate sequence of actions.

    ![Figure 5. Comparation of our methods with previous text-to-motion generation methods.](images/5.jpg)
    *该图像是比较不同文本到动作生成方法的图表。每一行展示了不同方法（如MDM、MotionCLIP、Codebook + Interpolation、AvatarCLIP、OOHMG）生成的动作姿态，并与我们的结果进行对比，显示出生成效果的多样性和真实性。*

## 6.2. Ablation Studies / Parameter Analysis
The paper conducts ablation studies to validate the effectiveness of the `Posture-Diffuser` and `Go-Diffuser` modules individually.

### 6.2.1. Posture-Diffuser Analysis
Figure 6 compares the poses generated by the `posture-diffuser` against other text-to-pose methods.
*   **Observation:** For prompts requiring precise body part control like "dance the waltz" or "kick soccer," methods like `OOHMG` and simple CLIP `Matching` fail to produce accurate poses. For more abstract prompts like "cry" or "pray," the `Matching` method generates identical, generic poses.
*   **Conclusion:** In contrast, PRO-Motion's method of using an LLM to generate a detailed, structured script allows the `posture-diffuser` to generate highly accurate and specific poses that correctly reflect the nuances of the prompt. This validates the "Plan -> Posture" pipeline.

    ![Figure 6. Comparison of our method with previous text-to-pose generation methods.](images/6.jpg)
    *该图像是一个对比图，展示了我们的方法与其他文本到姿态生成方法的效果。不同算法（如 Optimize, VPoserOpt, Matching, OOHMG 和我们的算法）针对相同的动作指令生成的姿态被清晰地展示，以便进行比较。*

### 6.2.2. Go-Diffuser Analysis
This study evaluates the `go-diffuser`'s ability to predict global motion from a sequence of static poses.

The following are the results from Table 2 of the original paper:

| Methods       | Average Positional Error ↓ | Average Variance Error ↓ |
|---------------|----------------------------|--------------------------|
|               | root joint                 | global traj.             | mean local | mean global | root joint                 | global traj.             | mean local | mean global |
| Regression    | 5.878673                   | 5.53344                  | 0.642252   | 5.919954    | 35.387340                  | 35.386562                | 0.147606   | 35.483219   |
| Baseline[62]  | 0.384152                   | 0.373394                 | 0.183978   | 0.469322    | 0.114308                   | 0.113845                 | 0.015207   | 0.126049   |
| **Ours**      | **0.365327**               | **0.354685**             | **0.128763**| **0.418265**| **0.111131**               | **0.110855**             | **0.008708**| **0.118334**|

**Analysis:**
*   **Quantitative Results (Table 2):** PRO-Motion's `go-diffuser` (`Ours`) achieves the lowest (best) error across all categories of `Average Positional Error` (APE) and `Average Variance Error` (AVE). This is particularly notable for `global traj.` (global trajectory) and `mean local` pose error, indicating that it is superior at both predicting the path of movement and reconstructing the body poses accurately.
*   **Qualitative Results (Figure 7):** The visualizations show that the proposed method produces motions with better fidelity. It captures fine details like knee flexion more accurately than the baselines and demonstrates a more nuanced understanding of how to generate motion trends even from very similar adjacent poses. This validates the design of the `go-diffuser`, especially its use of a transformer to treat key poses as discrete tokens.

    ![该图像是一个示意图，展示了不同方法生成的动作模型，包括GT、Reg.、B/L和我们的方案。不同的模型在动作表现上各有特点，显示出我们的方法在生成现实动作方面的优势。](images/7.jpg)
    *该图像是一个示意图，展示了不同方法生成的动作模型，包括GT、Reg.、B/L和我们的方案。不同的模型在动作表现上各有特点，显示出我们的方法在生成现实动作方面的优势。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully introduces **PRO-Motion**, a novel and effective "divide-and-conquer" framework for open-world text-to-motion generation. By decomposing the problem into three stages—planning with an LLM (`motion planner`), generating static poses from simple scripts (`posture-diffuser`), and synthesizing global dynamics (`go-diffuser`)—the model overcomes the key limitations of previous methods. It effectively handles ambiguous, open-world text prompts, generates temporally coherent pose sequences, and produces realistic, non-static motions. The experimental results demonstrate its state-of-the-art performance in both semantic alignment and motion quality.

## 7.2. Limitations & Future Work
The authors do not explicitly list limitations, but several can be inferred from the methodology:

*   **Dependence on LLM Quality:** The entire pipeline's success hinges on the `motion planner`. If the LLM produces a poor or illogical plan (sequence of scripts), the subsequent stages cannot recover, leading to a poor final motion. The quality is thus bound by the capabilities of the underlying LLM.
*   **Error Propagation:** The sequential, three-stage nature of the framework means that errors from an earlier stage can propagate and be amplified in later stages. A slightly incorrect pose from the `posture-diffuser` might lead to a very unnatural motion from the `go-diffuser`.
*   **Lack of End-to-End Training:** The modules are trained separately. An end-to-end differentiable framework could potentially allow the modules to co-adapt and achieve better overall performance, but this would be significantly more complex to design.
*   **Computational Cost:** The multi-stage process involving an LLM call and two separate diffusion models is likely more computationally expensive and slower at inference time compared to a single end-to-end model.

    Potential future research directions could include:
*   Exploring ways to make the pipeline end-to-end trainable.
*   Improving the motion planner, perhaps by fine-tuning a smaller LLM specifically for this task or allowing for interactive user feedback to refine the plan.
*   Extending the framework to more complex scenarios, such as generating motions involving human-object or human-human interactions.

## 7.3. Personal Insights & Critique
*   **Key Insight:** The most powerful idea in this paper is the use of an LLM as a **semantic decomposer**. It elegantly solves the open-vocabulary problem by translating any free-form text into a structured, simplified representation that a specialized model can easily handle. This paradigm of "LLM as a planner" is highly transferable and could be applied to other complex, cross-modal generation tasks (e.g., text-to-storyboarding, text-to-complex-scene-generation).

*   **Strengths:**
    *   The modular design is intellectually clean and makes the problem tractable. Each module has a clear, well-defined responsibility.
    *   It cleverly sidesteps the known weaknesses of using CLIP for motion generation (lack of temporal priors) by relying on an LLM's inherent sequential reasoning.
    *   The separation of local posture and global dynamics is a very effective way to ensure both pose accuracy and realistic movement through space.

*   **Potential Issues & Critique:**
    *   The "simplicity" of the posture scripts, while beneficial for the `posture-diffuser`, might also be a limitation. The five-rule system might filter out subtle stylistic or emotional nuances present in the original prompt that cannot be expressed through simple geometric relationships. For example, the difference between a "sad walk" and a "happy walk" might be lost.
    *   The evaluation, while thorough, could benefit from a user study to assess the perceived naturalness and semantic correctness of the generated motions, as metrics alone do not always capture the full picture of motion quality.
    *   The `Smooth` metric in Table 1 is not explained. The value for "Ours" is an order of magnitude higher than for other methods. While the arrow indicates higher is better, this large gap suggests it might be measuring a different quality of motion, and without a definition, it's hard to interpret its meaning fully. It could mean smoother, or it could mean slower, less dynamic motion. This ambiguity is a minor weakness in the paper's presentation.