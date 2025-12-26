# 1. Bibliographic Information
## 1.1. Title
Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference

The title clearly states the paper's core contributions:
1.  **Full Diffusion Trajectory Alignment:** The method is designed to optimize the entire generation process of a diffusion model, not just the final stages.
2.  **Fine-Grained Human Preference:** The alignment is not based on a fixed goal but can be controlled with detailed, specific preferences provided by a user.
3.  **Direct Alignment:** The method uses a direct optimization technique, likely involving backpropagation through a reward function, as opposed to indirect, policy-based reinforcement learning.

## 1.2. Authors
The authors are Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, and Yansong Tang.

Their affiliations include major research institutions and a leading tech company:
*   **Hunyuan, Tencent:** A large-scale AI research and development initiative at Tencent, focusing on foundational models, including text-to-image generation.
*   **The Chinese University of Hong Kong, Shenzhen (CUHK-Shenzhen):** A prominent research university.
*   **Shenzhen International Graduate School, Tsinghua University:** A graduate school of one of China's top universities.

    The collaboration between a major industry lab and top academic institutions suggests a strong blend of practical engineering and rigorous academic research.

## 1.3. Journal/Conference
The paper is available on arXiv, which is a preprint server. This means the paper has been shared publicly but has not yet undergone a formal peer-review process for publication in a conference or journal. The future publication date indicates its preprint status.

## 1.4. Publication Year
The publication date listed is September 8, 2025. This is a placeholder date for the arXiv submission, confirming its status as a recent preprint.

## 1.5. Abstract
The abstract summarizes the paper's core ideas. Current methods for aligning diffusion models with human preferences using differentiable rewards face two main issues: (1) they are computationally expensive because they rely on multi-step denoising, which limits optimization to only the last few steps of the diffusion process; and (2) they require extensive offline fine-tuning of reward models to achieve specific aesthetic qualities.

To solve these problems, the authors propose two innovations:
1.  **Direct-Align:** A method that allows for the accurate recovery of an original image from any noisy state in a single step. This is achieved by predefining a noise prior and using interpolation. This technique enables optimization across the *entire* diffusion trajectory and prevents over-optimization on the final steps.
2.  **Semantic Relative Preference Optimization (SRPO):** A new reward formulation where the reward signal is conditioned on text. This allows for *online* adjustment of the reward using positive and negative prompt words, reducing the need for offline reward model fine-tuning.

    By applying these methods to fine-tune the `FLUX.1.dev` model, the authors achieve a more than 3x improvement in human-evaluated realism and aesthetic quality.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2509.06942
*   **PDF Link:** https://arxiv.org/pdf/2509.06942v3.pdf
*   **Publication Status:** The paper is a preprint available on arXiv and has not yet been peer-reviewed or officially published in a conference or journal.

# 2. Executive Summary
## 2.1. Background & Motivation
The core problem this paper addresses is the challenge of efficiently and effectively aligning powerful text-to-image diffusion models with nuanced human preferences. While these models can generate stunning images, ensuring they consistently produce outputs that are aesthetically pleasing, realistic, and free of artifacts is a significant hurdle.

A popular approach is to use Reinforcement Learning from Human Feedback (RLHF), where a "reward model" trained on human preference data guides the fine-tuning of the diffusion model. Specifically, methods using **direct backpropagation** through a **differentiable reward model** have shown promise. However, they suffer from two critical limitations:

1.  **Limited and Inefficient Optimization:** Existing methods calculate the reward on an image generated after several denoising steps. Backpropagating the gradient from the reward through this multi-step process is computationally intensive and numerically unstable. Consequently, researchers are forced to restrict optimization to only the final few steps of the generation process. This narrow focus makes the model susceptible to **reward hacking**: it learns to generate images that "trick" the reward model into giving a high score but are actually low-quality (e.g., overly smooth, saturated, or lacking detail).

2.  **Inflexible and Costly Reward Systems:** Reward models are often biased or trained on limited criteria. To achieve specific aesthetic goals like photorealism or particular lighting, practitioners must perform costly offline adjustments, such as collecting new high-quality datasets to fine-tune the reward model itself. There is no easy "online" mechanism to adjust the reward signal during the alignment process.

    This paper's entry point is to tackle these two problems directly. The authors propose a way to make direct backpropagation stable enough to work on the *entire* diffusion trajectory and introduce a novel way to formulate the reward signal so it can be controlled on the fly using simple text prompts.

## 2.2. Main Contributions / Findings
The paper makes four key contributions:

1.  **A Novel Optimization Framework (`Direct-Align`) to Mitigate Reward Hacking:** The proposed `Direct-Align` method enables optimization across the full diffusion trajectory, not just the late stages. It achieves this by using a predefined noise prior to analytically recover a clean image from any noisy state in a single step. This avoids the instability of multi-step backpropagation and, combined with a discounting mechanism for late timesteps, effectively prevents the model from overfitting to the reward model's biases (reward hacking).

2.  **An Online Reward Adjustment Mechanism (`SRPO`):** The paper introduces **Semantic Relative Preference Optimization (SRPO)**. This technique reformulates the reward as a text-conditioned signal. By providing positive and negative "control words" in the prompt (e.g., "realistic photo" vs. "CG render"), SRPO can dynamically guide the optimization towards desired attributes and away from undesired ones. This significantly reduces the need for expensive offline fine-tuning of reward models.

3.  **State-of-the-Art Performance in Realism and Aesthetics:** When applied to the powerful `FLUX.1.dev` model, the proposed method demonstrates significant improvements. Human evaluations show a **3.7-fold increase in perceived realism** and a **3.1-fold increase in aesthetic quality** compared to the baseline model. To the authors' knowledge, this is the first work to systematically improve the realism of large-scale diffusion models without requiring new training data.

4.  **Breakthrough in Training Efficiency:** The proposed framework is remarkably efficient. The authors report that their method converges in just **10 minutes of training on 32 NVIDIA H20 GPUs**, a significant improvement over competing methods like `DanceGRPO`, which can take hundreds of GPU hours.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts

### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that learn to create data, such as images, by reversing a gradual noising process. The process has two parts:

*   **Forward Process (Noising):** This is a fixed process where a clean image, $x_0$, is gradually corrupted by adding a small amount of Gaussian noise at each of a series of timesteps, $T$. After $T$ steps, the image becomes indistinguishable from pure Gaussian noise. The state of the image at any timestep $t$ can be expressed directly as an interpolation between the original image and noise:
    \$
    x_t = \alpha_t x_0 + \sigma_t \epsilon
    \$
    where $\epsilon$ is a random Gaussian noise vector, and $\alpha_t$ and $\sigma_t$ are scheduling parameters that control the amount of original signal and noise at timestep $t$. As $t$ increases, $\alpha_t$ decreases and $\sigma_t$ increases.

*   **Reverse Process (Denoising):** The goal of the model is to learn to reverse this process. It takes a noisy image $x_t$ and a timestep $t$ as input and tries to predict the noise $\epsilon$ that was added to it. This learned model is denoted as $\epsilon_{\theta}(x_t, t, c)$, where $\theta$ are the model's parameters and $c$ is an optional condition, such as a text prompt. By iteratively subtracting the predicted noise, the model can generate a clean image $x_0$ starting from pure noise $x_T$.

### 3.1.2. Flow Matching Models (e.g., FLUX)
Flow Matching models are a more recent type of generative model that, like diffusion models, learn to transform a simple distribution (e.g., Gaussian noise) into a complex data distribution (e.g., images). Instead of a stochastic (random) noising process, they often learn a deterministic path described by an Ordinary Differential Equation (ODE). The model learns a vector field that "flows" points from the noise distribution to the image distribution. Models like `FLUX` are based on this principle and are known for being highly efficient, often requiring fewer steps to generate high-quality images compared to traditional diffusion models.

### 3.1.3. Reinforcement Learning from Human Feedback (RLHF)
RLHF is a technique used to align models with complex, subjective human goals. For text-to-image models, the process is:
1.  **Collect Human Preference Data:** Humans are shown pairs of images generated from the same prompt and asked to choose which one they prefer.
2.  **Train a Reward Model:** A model (the "reward model") is trained on this dataset to predict the human preference score for a given image and prompt.
3.  **Fine-tune the Generative Model:** The generative model is then fine-tuned using reinforcement learning. It generates an image, the reward model scores it, and this score (reward) is used to update the generative model's parameters to produce images that will receive higher scores in the future.

### 3.1.4. Differentiable Rewards and Reward Hacking
In the context of this paper, the reward model is **differentiable**, meaning we can compute the gradient of the reward score with respect to the input image. This allows for **direct backpropagation**: the gradient can flow from the reward model all the way back through the generation process to update the diffusion model's weights. This is generally more sample-efficient than policy-gradient RL methods.

**Reward Hacking** is a critical failure mode in this process. It occurs when the generative model learns to exploit flaws or biases in the reward model to maximize its score, even if it leads to poor-quality outputs. For example, if a reward model has a slight bias for brighter or more saturated images, the generative model might learn to produce unnaturally overexposed images because they achieve a high reward.

## 3.2. Previous Works
The paper situates its contributions relative to several lines of existing research:

*   **Direct Backpropagation Methods (`ReFL`, `DRaFT`):** These are the most direct predecessors. They fine-tune diffusion models by backpropagating gradients from a differentiable reward. However, as the paper notes, they face stability and cost issues. To get a reward, they must first generate an image. Backpropagating through the entire multi-step generation process is infeasible. So, they typically sample non-differentiably to a late timestep $x_k$ and then perform a few (or one) differentiable denoising steps. This limits the optimization to only the final stages of generation, where high-frequency details are refined, making the model prone to reward hacking on attributes like color and texture.

*   **Policy-Based RL Methods (`DPOK`, `GRPO`):** These methods treat the diffusion model as a policy and use policy gradient algorithms (like REINFORCE) to optimize it. While they can be more stable, they are often less sample-efficient than direct gradient methods. The paper specifically compares against `DanceGRPO`, a state-of-the-art method in this category, highlighting its own approach's superior training efficiency.

*   **Reward Model Refinement (`ICTHP`, `HPSv3`):** This line of work addresses reward hacking by improving the reward model itself. For instance, `ICTHP` collects a higher-quality dataset to fine-tune the reward model to be less biased. `HPSv3` trains reward models on generations from more advanced models. The current paper's approach is complementary; instead of changing the reward model offline, `SRPO` provides an *online* mechanism to control and regularize the *use* of an existing reward model during fine-tuning.

## 3.3. Technological Evolution
The field of text-to-image generation has evolved rapidly:
1.  **Early Generative Models (GANs, VAEs):** Initial attempts at image generation.
2.  **Rise of Diffusion Models (DDPM, DDIM):** These models brought a new level of quality and stability, leading to photorealistic generation.
3.  **Large-Scale Text-to-Image Models (DALL-E 2, Stable Diffusion):** Combining diffusion models with large language models (like CLIP) enabled powerful text-conditioned image generation.
4.  **Alignment with Human Preferences:** As models became more powerful, the focus shifted from raw capability to alignment—ensuring outputs are helpful, harmless, and aesthetically pleasing. This led to the application of RLHF.
5.  **Refining RLHF for Diffusion:** Early RLHF methods were often inefficient or unstable. Current research, including this paper, focuses on making the alignment process more robust, efficient, and controllable. This paper fits into this latest stage by proposing a solution to the key bottlenecks of direct gradient-based alignment.

## 3.4. Differentiation Analysis
*   **vs. `ReFL`/`DRaFT`:** The core difference is the optimization scope. `ReFL` and `DRaFT` are confined to **late timesteps** due to the instability of backpropagating through denoising. `Direct-Align` overcomes this with its **single-step image recovery** mechanism, enabling stable optimization across the **full trajectory**. This makes it fundamentally more robust to reward hacking.

*   **vs. `ICTHP`:** The difference is in the approach to reward quality. `ICTHP` improves the reward model **offline** by training it on better data. `SRPO` provides an **online** method to modulate and regularize an *existing* reward model during RL fine-tuning via prompt engineering. It's a method for using rewards better, not necessarily for building better rewards from scratch.

*   **vs. `DanceGRPO`:** The difference lies in the optimization algorithm and reward formulation. `DanceGRPO` uses less efficient **policy gradients** and relies on relative rewards between different generated samples. `SRPO` uses more efficient **direct gradients** and computes a semantic relative reward on a **single sample** by comparing its score under positive and negative text conditions.

# 4. Methodology
The paper's methodology is divided into two main components: `Direct-Align`, the framework for full-trajectory optimization, and `Semantic-Relative Preference Optimization (SRPO)`, the method for fine-grained reward control.

## 4.1. Principles
The guiding principle is to create an RL alignment framework that is both **robust** and **controllable**.
*   **Robustness** is achieved by enabling optimization over the entire diffusion process. Early timesteps determine the image's overall structure, while late timesteps refine details. By optimizing both, the model learns a more holistic representation of human preference, reducing the risk of reward hacking on superficial features.
*   **Controllability** is achieved by treating the reward not as a fixed black box but as a malleable function that can be steered by semantic inputs (text). This allows for dynamic, fine-grained control over the alignment process without needing to retrain the reward model.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Direct-Align: Enabling Full Trajectory Optimization

#### Limitations of Existing Approaches
The paper first identifies the weakness in prior direct backpropagation methods like `DRaFT` and `ReFL`. Their objectives can be generalized as:
1.  **DRaFT-like methods** perform multi-step differentiable sampling from an intermediate state $x_t$:
    \$
    r = R ( \mathrm { sample } ( \mathbf { x_ { t } } , \mathbf { c } ) )
    \$
2.  **ReFL-like methods** perform a single-step prediction of the clean image $x_0$ from $x_t$:
    \$
    r = R \left( \frac { \mathbf { x_ { t } } - \sigma _ { t } \epsilon _ { \theta } ( \mathbf { x_ { t } } , t , \mathbf { c } ) } { \alpha _ { t } } \right)
    \$
In these formulas:
*   $r$ is the calculated reward.
*   $R$ is the differentiable reward model.
*   $x_t$ is the noisy image at timestep $t$.
*   $c$ is the text condition.
*   $\epsilon_{\theta}(x_t, t, c)$ is the noise predicted by the diffusion model.
*   $\alpha_t$ and $\sigma_t$ are the noise schedule coefficients.

    The problem is that for early timesteps (when $t$ is small and the image is very noisy), the single-step prediction in `ReFL` is highly inaccurate, and the multi-step sampling in `DRaFT` is computationally explosive. Thus, both are restricted to late timesteps.

#### Single-Step Image Recovery: The Key Insight
`Direct-Align`'s innovation is a stable way to recover a clean image from *any* timestep. The method is based on the forward diffusion formula: $x_t = \alpha_t x_0 + \sigma_t \epsilon_{gt}$. This can be analytically inverted to find $x_0$:
$$
\mathbf { x_ { 0 } } = \frac { \mathbf { x_ { t } } - \sigma _ { t } \epsilon _ { g t } } { \alpha _ { t } }
$$
Here, $\epsilon_{gt}$ is the ground-truth Gaussian noise that was added. The authors propose a new optimization pipeline (visualized in Figure 2):
1.  Start with a clean image $x_0$.
2.  Inject a **known, predefined** ground-truth Gaussian noise prior $\epsilon$ to create the noisy state $x_t$.
3.  The model $\epsilon_\theta(x_t, t, c)$ then makes a prediction.
4.  The final image is reconstructed using a mix of the model's prediction and the known ground-truth noise. This is the crucial step for stability. Instead of relying entirely on the model's (potentially inaccurate) prediction, the paper uses it to refine a portion of the true noise. The reward is calculated on this more stable reconstruction:
    $$
    r = r \left( \frac { \mathbf { x_ { t } } - \Delta \sigma _ { t } \epsilon _ { \theta } ( \mathbf { x_ { t } } , t , \mathbf { c } ) - ( \sigma _ { t } - \Delta \sigma ) \epsilon } { \alpha _ { t } } \right)
    $$
*   **Explanation of Formula (Eq. 5):** The term $(\sigma_t \epsilon)$ from the standard inversion formula is split into two parts. A small part, $\Delta\sigma_t \epsilon_\theta$, comes from the model's prediction. The larger part, $(\sigma_t - \Delta\sigma)\epsilon$, comes from the *known ground-truth noise*. Because most of the denoising is done with the known truth, the reconstruction is highly accurate even at very noisy early timesteps. This avoids gradient explosion and allows for effective optimization across the full trajectory. Figure 3 in the paper visually demonstrates this superior reconstruction quality.

    The following diagram illustrates the `Direct-Align` optimization pipeline.

    ![该图像是示意图，展示了直接对齐全扩散轨迹与细粒度人类偏好的方法。图中包含两个主要部分：Direct-Align 和语义相对偏好优化（SRPO）。Direct-Align部分说明了如何将高斯噪声注入图像并通过梯度去噪及逆高斯方法恢复图像，配合奖励模型（RM）进行评估。SRPO部分则描述了如何根据正负提示动态调整奖励与惩罚，从而优化人类评价的现实感和美学质量。](images/2.jpg)
    *该图像是示意图，展示了直接对齐全扩散轨迹与细粒度人类偏好的方法。图中包含两个主要部分：Direct-Align 和语义相对偏好优化（SRPO）。Direct-Align部分说明了如何将高斯噪声注入图像并通过梯度去噪及逆高斯方法恢复图像，配合奖励模型（RM）进行评估。SRPO部分则描述了如何根据正负提示动态调整奖励与惩罚，从而优化人类评价的现实感和美学质量。*

#### Reward Aggregation Framework
To further improve stability and prevent overfitting to late timesteps, the method aggregates rewards over a small sequence of timesteps. A decaying discount factor $\lambda(t)$ is applied, which gives less weight to rewards from later timesteps, where reward hacking is more prevalent.
$$
r ( { \bf x _ { t } } ) = \lambda ( t ) \cdot \sum _ { k } ^ { k - n } r ( x _ { i } - \epsilon _ { \theta } ( { \bf x _ { i } } , i , { \bf c } ) , { \bf c } )
$$
*   **Explanation of Formula (Eq. 2):** This formula describes the total reward signal for an update. $\lambda(t)$ is a weight that decreases as the timestep $t$ gets later (closer to a clean image). The sum aggregates rewards from several consecutive steps, creating a more stable gradient signal.

### 4.2.2. Semantic-Relative Preference Optimization (SRPO)

#### Semantic Guided Preference (SGP)
The first step of `SRPO` is to reframe the reward as a controllable, text-conditioned function. A standard reward model (like `HPSv2`) computes a score based on the similarity between an image embedding and a text embedding, following the CLIP architecture:
$$
r ( \mathbf { x } ) = R M ( \mathbf { x } , \mathbf { p } ) \propto f _ { i m g } ( \mathbf { x } ) ^ { T } \cdot f _ { t x t } ( \mathbf { p } )
$$
*   **Explanation of Formula (Eq. 6):** $f_{img}(\mathbf{x})$ is the feature vector of the image $\mathbf{x}$, and $f_{txt}(\mathbf{p})$ is the feature vector of the prompt $\mathbf{p}$. The reward $r(\mathbf{x})$ is proportional to the dot product (cosine similarity) of these two vectors.

    The key insight of SGP is that by augmenting the prompt $\mathbf{p}$ with "control words" $\mathbf{p_c}$ (e.g., "realistic photo," "cinematic lighting"), one can steer the text embedding and thus the preference of the reward model. The SGP reward is:
$$
r _ { S G P } ( \mathbf { x } ) = R M ( \mathbf { x } , ( \mathbf { p _ { c } } , \mathbf { p } ) ) \propto f _ { i m g } ( \mathbf { x } ) ^ { T } \cdot \mathbf { C _ { ( p _ { c } , p ) } }
$$
*   **Explanation of Formula (Eq. 7):** Here, $\mathbf{C_{(p_c, p)}}$ is the new text embedding generated from the augmented prompt. This allows for online control of the reward's characteristics.

#### Semantic-Relative Preference
While SGP provides control, it doesn't solve the problem of inherent biases in the reward model (e.g., a preference for reddish tones). `SRPO` addresses this by calculating a **relative** reward. For a single image $\mathbf{x}$, two reward scores are computed: one with a positive/desired prompt ($C_1$, e.g., "realistic photo, a dog") and one with a negative/undesired prompt ($C_2$, e.g., "cartoon, a dog"). The optimization objective is the difference between them:
$$
\begin{array} { r l } & { r _ { S R P } ( \mathbf { x } ) = r _ { 1 } - r _ { 2 } } \\ & { ~ = f _ { i m g } ( \mathbf { x } ) ^ { T } \cdot ( \mathbf { C } _ { 1 } - \mathbf { C } _ { 2 } ) } \end{array}
$$
*   **Explanation of Formulas (Eq. 8, 9):** This formulation is powerful. Any bias inherent in the image encoder $f_{img}(\mathbf{x})$ that is independent of the text prompt's desired semantic change will be present in both $r_1$ and $r_2$. By taking the difference, these common biases are effectively regularized or cancelled out. The optimization is forced to focus only on the semantic difference between $C_1$ and $C_2$, leading to more targeted alignment and less reward hacking.

    The paper also proposes an alternative formulation inspired by classifier-free guidance, which interpolates between the negative and positive embeddings:
$$
r _ { C F G } ( \mathbf { x } ) = f _ { i m g } ( \mathbf { x } ) ^ { T } \cdot ( ( 1 - k ) \cdot \mathbf { C } _ { 2 } + k \cdot \mathbf { C } _ { 1 } )
$$
*   **Explanation of Formula (Eq. 10):** Here, $k$ is a scaling factor. This formula provides another way to balance the influence of desired and undesired attributes.

#### Inversion-Based Regularization
`Direct-Align`'s unique structure enables another powerful regularization technique. Because the image reconstruction is decoupled from the model's prediction, the framework supports optimization in both the **denoising** (image-improving) and **inversion** (image-degrading) directions. `SRPO` leverages this by applying the positive reward during denoising and the negative reward during inversion.
$$
\begin{array} { l } { r _ { 1 } = r _ { 1 } \left( \frac { \mathbf { a } - \Delta \sigma _ { t } \epsilon _ { \theta } \left( \mathbf { x _ { t } } , t , \mathbf { c } \right) } { \alpha _ { t } } \right) } \\ { r _ { 2 } = r _ { 2 } \left( \frac { \mathbf { b } + \Delta \sigma _ { t } \epsilon _ { \theta } \left( \mathbf { x _ { t } } , t, \mathbf { c } \right) } { \alpha _ { t } } \right) } \end{array}
$$
*   **Explanation of Formulas (Eq. 12, 13):** The paper is slightly opaque here, but the logic is as follows. $r_1$ represents the reward for denoising (gradient ascent), where the model's prediction $\epsilon_\theta$ is subtracted. $r_2$ represents the penalty for inversion (gradient descent), where the model's prediction is *added*, effectively moving the image *away* from the clean version. $r_1$ would be associated with the positive prompt and $r_2$ with the negative prompt. This decouples the reward and penalty terms, applying them at different timesteps or in different directions to enhance optimization robustness.

# 5. Experimental Setup
## 5.1. Datasets
*   **Training and Evaluation Dataset:** `Human Preference Dataset v2 (HPDv2)`. This dataset contains human preference labels for images generated from prompts covering four visual concepts from `DiffusionDB`. The evaluation is performed on the benchmark's test set of 3,200 prompts.
*   **Reason for Choice:** `HPDv2` is a standard and widely used benchmark for evaluating the alignment of text-to-image models with human preferences, making it suitable for comparing against other state-of-the-art methods.

## 5.2. Evaluation Metrics
The paper uses a comprehensive set of automatic and human evaluation metrics.

### 5.2.1. Automatic Metrics
*   **`Aesthetic Score v2.5`:**
    1.  **Conceptual Definition:** An automated predictor trained to score the aesthetic quality of an image on a scale, typically from 1 to 10. It is trained on a large dataset of images with aesthetic ratings (`LAION-Aesthetics`). A higher score indicates better visual appeal.
    2.  **Mathematical Formula:** It's a deep learning model, so there isn't a simple formula. It's typically a regression model $f_{aes}(\mathbf{x})$ that outputs a scalar score: $Score = f_{aes}(\mathbf{x})$.

*   **`PickScore`:**
    1.  **Conceptual Definition:** A reward model trained on the `Pick-a-Pic` dataset, which contains millions of user preferences between pairs of generated images. It is designed to reflect human choices regarding both text-image alignment and overall quality. A higher score is better.
    2.  **Mathematical Formula:** Like `Aesthetic Score`, it's a deep model. It takes an image $\mathbf{x}$ and a prompt $\mathbf{p}$ and outputs a preference score: $Score = f_{pick}(\mathbf{x}, \mathbf{p})$.

*   **`ImageReward`:**
    1.  **Conceptual Definition:** Another reward model trained on human preferences, specifically designed to evaluate text-to-image generation. It aims to score images based on multiple criteria like prompt alignment and visual quality. A higher score is better.
    2.  **Mathematical Formula:** A deep model that computes a score based on an image and prompt: $Score = f_{ir}(\mathbf{x}, \mathbf{p})$.

*   **`HPSv2.1` (Human Preference Score v2.1):**
    1.  **Conceptual Definition:** The reward model used for training in the experiments. It is also trained on human preference data (`HPDv2`) and is designed to predict which image a human would prefer. A higher score indicates a higher likelihood of being preferred by humans.
    2.  **Mathematical Formula:** A deep model that computes a preference score: $Score = f_{hps}(\mathbf{x}, \mathbf{p})$.

*   **`SGP-HPS` (Semantic Guided Preference - HPS):**
    1.  **Conceptual Definition:** A custom metric proposed by the authors to specifically measure the model's ability to generate realistic images. It quantifies the difference in `HPSv2.1` score when the prompt is augmented with "Realistic photo" versus "CG Render." A larger positive difference indicates the model is better at producing images that align with the concept of "realism" as understood by the reward model.
    2.  **Mathematical Formula:**
        \$
        SGP-HPS = \text{HPSv2.1}(\mathbf{x}, \text{"Realistic photo, "} \mathbf{p}) - \text{HPSv2.1}(\mathbf{x}, \text{"CG Render, "} \mathbf{p})
        \$
    3.  **Symbol Explanation:** $\mathbf{x}$ is the generated image, and $\mathbf{p}$ is the original prompt.

*   **`GenEval`:**
    1.  **Conceptual Definition:** An evaluation framework that focuses on object-level text-to-image alignment. It measures whether the key objects mentioned in the prompt are correctly generated in the image.
    2.  **Mathematical Formula:** It involves object detection and CLIP-based scoring, so a single formula is not representative. The final score is an aggregation of object presence and correctness scores.

*   **`DeQA` (Degradation-aware Quality Assessment):**
    1.  **Conceptual Definition:** An image quality assessment metric designed to detect common AI-generated artifacts and degradations. A lower score typically indicates fewer artifacts and better quality.

### 5.2.2. Human Evaluation
*   **Protocol:** A rigorous study was conducted with 10 trained annotators and 3 domain experts on 500 prompts. Each prompt was evaluated by 5 different annotators.
*   **Dimensions:** Images were rated on a four-level ordinal scale (Excellent, Good, Pass, Fail) across four dimensions:
    1.  **Text-image alignment**
    2.  **Realism and artifact presence**
    3.  **Detail complexity and richness**
    4.  **Aesthetic composition and appeal**
        An "Overall Quality" score was also derived.

## 5.3. Baselines
The proposed method (`SRPO` built on `Direct-Align`) is compared against several representative baselines:
*   **`FLUX.1.dev`:** The base model before any RL fine-tuning. This is the starting point and serves as the main reference.
*   **`ReFL` and `DRaFT-LV`:** State-of-the-art direct backpropagation methods. These are crucial baselines as the paper's `Direct-Align` framework is designed to overcome their limitations.
*   **`DanceGRPO`:** A state-of-the-art policy-based RL alignment method. This baseline represents the alternative paradigm to direct backpropagation.
*   **`FLUX.1.Krea`:** A more recent, improved version of the `FLUX` model, serving as a very strong baseline to show the effectiveness of the proposed fine-tuning method.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The experimental results strongly validate the effectiveness of the `SRPO` framework.

The main results are presented in Table 1. A key takeaway is the stark contrast between automatic reward scores and human evaluation. While methods like `ReFL` and `DanceGRPO` show slight improvements in some automatic metrics like `HPS` and `ImageReward`, their human evaluation scores for `Realism` and `Aesthetics` are either on par with or *worse* than the original `FLUX` baseline. This is a classic sign of **reward hacking**—the models got better at pleasing the automatic scorer but not actual humans.

In contrast, **`SRPO` achieves a massive leap in human-evaluated quality**. The "Excellent" rate for Realism jumps from 8.2% for the baseline to **38.9%** for `SRPO`. The Overall preference "Excellent" rate soars from 5.3% to **29.4%**. This demonstrates that `SRPO` successfully aligns the model with genuine human preference for realism and aesthetics while avoiding reward hacking.

Another crucial result is **training efficiency**. `SRPO` required only **5.3 GPU hours**, while `DanceGRPO`, a powerful policy-based method, took **480 GPU hours**. This is nearly a **100x speedup**, making the proposed method highly practical.

The visual comparisons in the paper support these numbers. Figure 4 shows that `SRPO` generates images with significantly fewer AI artifacts and more natural textures compared to other methods. Figure 5 highlights `SRPO`'s superior performance in detail and realism compared to `FLUX` and `DanceGRPO`.

## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="5">Reward</th>
<th colspan="2">Other Metrics</th>
<th colspan="3">Human Eval (Excellent Rate %)</th>
<th rowspan="2">GPU hours(H20)</th>
</tr>
<tr>
<th>Aes</th>
<th>Pick</th>
<th>ImageReward</th>
<th>HPS</th>
<th>SGP-HPS</th>
<th>GenEval</th>
<th>DeQA</th>
<th>Real</th>
<th>Aesth</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td>FLUX</td>
<td>5.867</td>
<td>22.671</td>
<td>1.115</td>
<td>0.289</td>
<td>0.463</td>
<td>0.678</td>
<td>4.292</td>
<td>8.2</td>
<td>9.8</td>
<td>5.3</td>
<td>−</td>
</tr>
<tr>
<td>ReFL*</td>
<td>5.903</td>
<td>22.975</td>
<td>1.195</td>
<td>0.298</td>
<td>0.470</td>
<td>0.656</td>
<td>4.299</td>
<td>5.5</td>
<td>6.6</td>
<td>3.1</td>
<td>16</td>
</tr>
<tr>
<td>DRaFT-LV*</td>
<td>5.729</td>
<td>22.932</td>
<td>1.178</td>
<td>0.296</td>
<td>0.458</td>
<td>0.636</td>
<td>4.236</td>
<td>8.3</td>
<td>9.7</td>
<td>4.7</td>
<td>24</td>
</tr>
<tr>
<td>DanceGRPO</td>
<td>6.022</td>
<td>22.803</td>
<td>1.218</td>
<td>0.297</td>
<td>0.414</td>
<td>0.585</td>
<td>4.353</td>
<td>5.3</td>
<td>8.5</td>
<td>3.7</td>
<td>480</td>
</tr>
<tr>
<td>Direct-Align</td>
<td>6.032</td>
<td>23.030</td>
<td>1.223</td>
<td>0.294</td>
<td>0.448</td>
<td>0.668</td>
<td>4.373</td>
<td>5.9</td>
<td>8.7</td>
<td>3.9</td>
<td>16</td>
</tr>
<tr>
<td>SRPO</td>
<td>6.194</td>
<td>23.040</td>
<td>1.118</td>
<td>0.289</td>
<td>0.505</td>
<td>0.665</td>
<td>4.275</td>
<td>38.9</td>
<td>40.5</td>
<td>29.4</td>
<td>5.3</td>
</tr>
</tbody>
</table>

*(Note: `*` indicates implementation by the authors.)*

## 6.3. Ablation Studies / Parameter Analysis

The paper includes several insightful analyses to dissect why their method works.

*   **Denoising Efficiency (Figure 3):** This experiment compares the image recovery quality of standard one-step prediction vs. the `Direct-Align` method at very early, noisy timesteps. The results are striking: `Direct-Align` can recover a coherent image structure even from a state with 95% noise, whereas standard prediction fails completely. This visually confirms the effectiveness of the single-step recovery mechanism, which is the foundation for full-trajectory optimization.

    The comparison of one-step prediction quality is shown below.

    ![Figure 3. Comparison on one-step prediction at early timestep The values 0.075 and 0.025 denote the weight of the model prediction term used for method, respectively. The earliest $5 \\%$ represent state with $9 5 \\%$ noise from an unshifted timestep. By constructing a Gaussian prior, our one-step sampling method achieves highquality results at early timesteps, even when the input image is highly noised.](images/3.jpg)
    *该图像是一个比较图表，展示了不同时间步长下的图像预测质量。上方为先前方法的结果，下方是我们的方法在两种不同权重参数（0.075和0.025）下的效果。每一列代表不同的噪声水平，从最早的5%到50%。该方法通过构建高斯先验，实现了在早期时间步长下的高质量图像重建。*

*   **Optimization Timestep (Figure 7):** This study directly tests the paper's central hypothesis about reward hacking. By training the model on different intervals of the diffusion trajectory (Early, All, Late), they show that training exclusively on **late timesteps leads to a massive 77% "hacking rate"** in human evaluation. This provides strong evidence that restricting optimization to late stages, as prior methods do, is a primary cause of reward hacking. `Direct-Align`'s ability to train on the full trajectory is thus a critical advantage.

*   **Effectiveness of `Direct-Align` Components (Figure 9d):** An ablation study on `SRPO` shows that removing key components of the `Direct-Align` framework—namely **early timestep optimization** and the **late-timestep discount factor $\lambda(t)$**—leads to a significant drop in performance and a resurgence of reward hacking artifacts like oversaturation. This confirms that both elements are essential for the method's success.

*   **Fine-Grained Control with `SRPO` (Figure 8):** This section demonstrates the "fine-grained preference" aspect of the method. By adding simple control words like "dark," "bright," "comic," or "Renaissance" to the prompts during `SRPO` training, the model learns to generate images in the corresponding style. Human evaluations (Figure 9c) confirm that this styling is effective. This highlights the flexibility of `SRPO` in steering the model towards diverse aesthetic goals online, without needing to retrain the reward model.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces a novel and highly effective reinforcement learning framework (`Direct-Align` + `SRPO`) for aligning text-to-image models with fine-grained human preferences. The work successfully tackles two major challenges in the field:

1.  It overcomes the computational bottleneck of direct gradient methods by proposing `Direct-Align`, a technique that enables stable optimization over the **full diffusion trajectory**. This fundamentally mitigates the problem of **reward hacking** that plagued previous methods limited to late-stage optimization.
2.  It introduces `SRPO`, a flexible reward formulation that allows for **online adjustment of aesthetic preferences** using simple text-based control words. This semantic-relative approach effectively regularizes against reward model biases and reduces the reliance on costly offline reward model fine-tuning.

    The method achieves state-of-the-art results, dramatically improving the realism and aesthetic quality of the `FLUX` model as judged by human evaluators, all while being exceptionally training-efficient.

## 7.2. Limitations & Future Work
The authors candidly discuss the limitations of their current work and point to future research directions:

*   **Limitations:**
    1.  **Controllability Dependency:** The effectiveness of `SRPO`'s control mechanism is dependent on the underlying reward model's ability to understand the "control words." If a style or concept is too far outside the reward model's training distribution, the control will be less effective.
    2.  **Interpretability:** The mapping of control words to changes in the latent space is not always perfectly aligned with human intuition. The precise effect of a control word is mediated by the text encoder and may not be fully predictable.

*   **Future Work:**
    1.  **Systematic Control Strategy:** The authors plan to develop more systematic control strategies, potentially involving learnable tokens instead of fixed text words.
    2.  **Responsive Reward Model:** A key future direction is to fine-tune a vision-language model (VLM) to be explicitly responsive to a system of control words, creating a reward model that is co-designed with the `SRPO` framework.
    3.  **Generalization:** The authors suggest that the `SRPO` framework could be extended to other online RL algorithms, including those that work with non-differentiable rewards.

## 7.3. Personal Insights & Critique
This paper presents a very well-designed and impactful piece of research.

*   **Key Strengths:**
    *   **Elegant Problem-Solving:** The `Direct-Align` method is a particularly clever solution to the long-standing problem of stability in direct reward optimization. Using the analytical inverse of the diffusion forward process with a known noise prior is an elegant way to sidestep the intractable backpropagation through a multi-step sampler.
    *   **Practicality and Efficiency:** The massive improvement in training efficiency (10 minutes vs. hundreds of hours) is a significant engineering contribution. It makes sophisticated alignment techniques accessible to a much broader range of researchers and practitioners.
    *   **Focus on Real Human Preference:** The paper's emphasis on human evaluation and its success in improving metrics like "realism" is crucial. It moves the field away from simply optimizing proxy automatic scores (which leads to reward hacking) and towards generating images that people genuinely find better. The `SRPO` mechanism is a direct and effective tool for this.

*   **Potential Critiques and Further Thoughts:**
    *   **Dependence on Reward Model:** As the authors note, the method is still fundamentally tethered to the capabilities of the base reward model. The `SRPO` technique is a clever way to "debias" or "steer" a given reward model, but it cannot invent knowledge the reward model does not possess. This highlights the continued importance of developing better and more robust reward models in parallel.
    *   **"Inversion-Based Regularization":** This is one of the most intriguing but least-explored ideas in the paper. The concept of using the "noising" direction for regularization feels akin to contrastive learning. A more detailed theoretical and empirical analysis of this specific component would be highly valuable. Does it work by penalizing low-density areas in the learned distribution? Does it improve mode coverage?
    *   **Blurring SFT and RL:** The finding that `SRPO` can work with offline real-world photographs is fascinating. It suggests the framework acts as a hybrid of Supervised Fine-Tuning (SFT) and Reinforcement Learning. It's not just fitting to data (like SFT) because it still uses the preference signal from the reward model. Exploring this hybrid nature could lead to new, even more powerful alignment paradigms.