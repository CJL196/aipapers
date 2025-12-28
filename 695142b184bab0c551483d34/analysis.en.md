# 1. Bibliographic Information

## 1.1. Title
**DanceGRPO: Unleashing GRPO on Visual Generation**

## 1.2. Authors
The paper is authored by Zeyue Xue (ByteDance Seed, The University of Hong Kong), Jie Wu (ByteDance Seed), Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. The authors hail from prominent research institutions and industry labs, specifically ByteDance's Seed team and The University of Hong Kong, with expertise in generative modeling and reinforcement learning.

## 1.3. Journal/Conference
This paper was published as a preprint on **arXiv** (v4 updated in May 2025). While not yet appearing in a peer-reviewed journal at the time of this writing, the research stems from highly reputable institutions known for state-of-the-art AI research (ByteDance and HKU).

## 1.4. Publication Year
The most recent version (v4) was published on **May 12, 2025**.

## 1.5. Abstract
The abstract highlights a critical challenge in generative AI: aligning visual model outputs with human preferences. While Reinforcement Learning (RL) is a promising fine-tuning approach, existing methods like `DDPO` and `DPOK` are unstable when scaled to large prompt sets. The paper introduces **DanceGRPO**, a framework adapting **Group Relative Policy Optimization (GRPO)** for visual generation. By reformulating sampling as **Stochastic Differential Equations (SDEs)**, DanceGRPO provides stable optimization across diffusion models and rectified flows. Experiments show performance gains of up to $181\%$ on benchmarks like `HPS-v2.1` and `VideoAlign`, demonstrating versatility across text-to-image (T2I), text-to-video (T2V), and image-to-video (I2V) tasks.

## 1.6. Original Source Link
*   **Original Source Link:** [https://arxiv.org/abs/2505.07818](https://arxiv.org/abs/2505.07818)
*   **PDF Link:** [https://arxiv.org/pdf/2505.07818v4.pdf](https://arxiv.org/pdf/2505.07818v4.pdf)

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
Generative models, such as **Diffusion Models** and **Rectified Flows**, have reached impressive levels of visual fidelity. However, "good" images are not always "aligned" images. Alignment refers to making the model follow human preferences (e.g., aesthetics, specific prompt adherence). 

**The Problem:**
1.  **Instability:** Previous RL methods like `DDPO` (Denoising Diffusion Policy Optimization) work on small datasets but break down (diverge) when training on thousands of diverse prompts.
2.  **VRAM Inefficiency:** Methods like `ReFL` require "differentiable reward models," meaning the model must calculate gradients through the reward function, which consumes massive amounts of GPU memory, especially for video.
3.  **Paradigm Conflict:** New models use **Rectified Flows** (deterministic paths), which conflict with the stochastic (random) nature required for traditional Reinforcement Learning.

**The Innovation:** 
The authors leverage **GRPO**, a method recently popularized by **DeepSeek-R1** for language models. GRPO eliminates the need for a separate "Value Model" (which predicts rewards) by using relative rewards within a group of samples. DanceGRPO is the first to adapt this "group-thinking" to the visual domain.

## 2.2. Main Contributions / Findings
1.  **Discovery of Stability:** The paper identifies that GRPO’s inherent stability mechanisms solve the core optimization failures seen in previous visual RL attempts.
2.  **SDE Reformulation:** It provides a mathematical bridge allowing Rectified Flow models to be trained with RL by turning their deterministic paths into **Stochastic Differential Equations (SDEs)**.
3.  **Unified Framework:** DanceGRPO is shown to be a "one-size-fits-all" solution, working seamlessly across images and videos, and across different model architectures (Stable Diffusion, FLUX, HunyuanVideo).
4.  **High Performance:** It achieves massive improvements in motion quality ($+181\%$) and aesthetic scores across standard benchmarks.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Diffusion Models & Rectified Flows
*   **Diffusion Models:** These models learn to generate data by reversing a "noise" process. They start with pure noise and iteratively remove it to reveal an image.
*   **Rectified Flows:** A newer paradigm where the model learns a straight "velocity" path from noise to data. It is often faster and more efficient than standard diffusion.

### 3.1.2. Reinforcement Learning from Human Feedback (RLHF)
RLHF is a three-step process:
1.  **Pretraining:** The model learns to generate general images.
2.  **Reward Modeling:** A separate model (the "judge") is trained to score images based on human preferences.
3.  **Fine-tuning (Alignment):** The generative model is trained to maximize the score from the judge.

### 3.1.3. Markov Decision Process (MDP)
For RL to work, the task must be framed as a sequence of steps where an **Agent** (the model) takes an **Action** (denoising a step) based on a **State** (the current noisy image) to receive a **Reward** (the final aesthetic score).

## 3.2. Previous Works
*   **DDPO (Denoising Diffusion Policy Optimization):** The first major work to apply policy gradients to diffusion. It treats the entire denoising chain as an RL trajectory. However, it requires a "Value Function" or global statistics that become unstable as prompts become more diverse.
*   **DPO (Direct Preference Optimization):** A method that avoids RL by using a classification-like loss on pairs of "better" vs "worse" images. While stable, the authors note it often yields only marginal quality improvements compared to pure RL.
*   **DeepSeek-R1 & GRPO:** In language modeling, GRPO was introduced to save memory. Instead of a Critic model (another large neural network), it uses the mean and standard deviation of rewards from a group of $G$ outputs to determine which outputs are "relatively" better.

## 3.3. Technological Evolution
The field moved from **Supervised Fine-Tuning (SFT)** (copying human examples) to **RLHF** (optimizing for scores). In visual generation, the shift is now from **Differentiable Rewards** (limited by VRAM) to **Black-box RL** (which can use any judge, even humans), with DanceGRPO providing the stability needed for that jump.

---

# 4. Methodology

## 4.1. Principles
The core idea of **DanceGRPO** is to treat the denoising process as a "trajectory" in Reinforcement Learning. To do this, it must solve two things:
1.  **Introduce Randomness:** RL needs to "explore" different versions of an image to see which one gets a higher reward.
2.  **Group Optimization:** Use the relative scores of these versions to update the model.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Denoising as a Markov Decision Process (MDP)
First, the authors define the denoising steps of a model as an RL environment. A state $s_t$ consists of the prompt $c$, the current timestep $t$, and the noisy latent $z_t$. The action $a_t$ is the next latent $z_{t-1}$. The reward $R$ is only given at the very end ($t=0$) based on the final image quality.

The sampling process is governed by a policy $\pi(a_t | s_t)$, which is the probability of moving from $z_t$ to $z_{t-1}$ given the prompt.

### 4.2.2. Formulation of Sampling SDEs
To allow for RL exploration, the authors must ensure the sampling is stochastic (has randomness). 

**For Diffusion Models:**
They use the reverse **Stochastic Differential Equation (SDE)**:
\$
\mathrm{d}\mathbf{z}_t = \left( f_t \mathbf{z}_t - \frac{1 + \varepsilon_t^2}{2} g_t^2 \nabla \log p_t (\mathbf{z}_t) \right) \mathrm{d}t + \varepsilon_t g_t \mathrm{d}\mathbf{w}
\$
*   $\mathbf{z}_t$: The image at time $t$.
*   $f_t, g_t$: Variables defining the noise schedule (how noise is added/removed).
*   $\nabla \log p_t (\mathbf{z}_t)$: The "score function" or the direction toward "clean" data.
*   $\varepsilon_t$: A hyperparameter that controls how much randomness (stochasticity) is added during sampling.
*   $\mathrm{d}\mathbf{w}$: Brownian motion (random noise).

**For Rectified Flows:**
Since Rectified Flows are naturally deterministic (ODEs), the authors introduce a new SDE version for the reverse process:
\$
\mathrm{d}\mathbf{z}_t = (\mathbf{u}_t - \frac{1}{2} \varepsilon_t^2 \nabla \log p_t (\mathbf{z}_t)) \mathrm{d}t + \varepsilon_t \mathrm{d}\mathbf{w}
\$
*   $\mathbf{u}_t$: The "velocity" or predicted direction from the model.
*   $\varepsilon_t$: Again, the noise level that introduces exploration.

    This allows the model to "try" different paths to a clean image for the same prompt.

### 4.2.3. The DanceGRPO Algorithm
Once the model samples a group of $G$ outputs $\{\mathbf{o}_1, \mathbf{o}_2, \dots, \mathbf{o}_G\}$ for a single prompt $c$, it calculates the **Advantage Function** $A_i$ for each output $i$.

The advantage is calculated as:
\$
A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \dots, r_G\})}{\mathrm{std}(\{r_1, r_2, \dots, r_G\})}
\$
*   $r_i$: The reward score for output $i$.
*   $\mathrm{mean}, \mathrm{std}$: The average and spread of rewards in that specific group.

    The model is then updated by maximizing the objective function $\mathcal{I}(\theta)$:
\$
\mathcal{I}(\theta) = \mathbb{E}_{\{\mathbf{o}_i\}_{i=1}^G \sim \pi_{\theta_{\mathrm{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=1}^T \min \left( \rho_{t,i} A_i, \mathrm{clip}(\rho_{t,i}, 1-\epsilon, 1+\epsilon) A_i \right) \right]
\$
*   $\rho_{t,i}$: The "importance sampling ratio" $\frac{\pi_{\theta}(a_{t,i}|s_{t,i})}{\pi_{\theta_{\mathrm{old}}}(a_{t,i}|s_{t,i})}$, which measures how much the current model $\theta$ has changed from the version that collected the data $\theta_{\mathrm{old}}$.
*   $\mathrm{clip}$: A safety mechanism that prevents the model from changing too drastically in one step (standard in PPO-style algorithms).
*   $\epsilon$: A small hyperparameter for the clip range.

### 4.2.4. Practical Implementation Details
*   **Shared Initialization Noise:** Unlike previous methods that used different random noise for every sample, DanceGRPO uses the **same** starting noise for all $G$ samples in a group. This makes the reward comparison purely about the model's choices, not the luck of the starting noise.
*   **Timestep Selection:** The authors found that they don't need to train on all 50 or 100 denoising steps. Training on a random $60\%$ of steps (or even just the first $30\%$) is often sufficient and much faster.

    The following figure (Figure 5 from the original paper) demonstrates the stability of this approach compared to the previous state-of-the-art:

    ![Figure 5 We visualize the results of DDPO and Ours. DDPO always diverges when applied to rectified fow SDEs](images/5.jpg)
    *Figure 5: DanceGRPO maintains high rewards while DDPO diverges and fails when applied to Rectified Flow SDEs.*

---

# 5. Experimental Setup

## 5.1. Datasets
The authors used large-scale datasets (over 10,000 prompts) to prove scalability:
*   **Text-to-Image:** Stable Diffusion v1.4, FLUX.1-dev, and HunyuanVideo-T2I.
*   **Text-to-Video:** HunyuanVideo using prompts from `VidProM` (a million-scale video prompt dataset).
*   **Image-to-Video:** SkyReels-I2V using `ConsisID` for identity-preserving video generation.

## 5.2. Evaluation Metrics

### 5.2.1. HPS-v2.1 (Human Preference Score)
1.  **Conceptual Definition:** A model trained to predict which of two images a human would prefer based on aesthetic beauty and prompt alignment.
2.  **Mathematical Formula:** Usually represented as a logit output from a Vision-Transformer:
    $s = \mathrm{MLP}(\mathrm{ViT}(x))$
3.  **Symbol Explanation:** $x$ is the generated image, $\mathrm{ViT}$ is the vision backbone, and $s$ is the scalar score.

### 5.2.2. CLIP Score
1.  **Conceptual Definition:** Measures how well the image matches the text prompt by looking at the cosine similarity between their embeddings.
2.  **Mathematical Formula:**
    \$
    \mathrm{CLIP}(I, T) = \frac{E_I \cdot E_T}{\|E_I\| \|E_T\|}
    \$
3.  **Symbol Explanation:** $E_I$ is the image embedding vector; $E_T$ is the text embedding vector.

### 5.2.3. VideoAlign
1.  **Conceptual Definition:** A multi-dimensional metric for video, specifically assessing **Visual Quality (VQ)**, **Motion Quality (MQ)**, and **Text-Alignment (TA)**.

## 5.3. Baselines
*   **DDPO / DPOK:** Standard policy gradient RL methods for diffusion.
*   **ReFL:** A method using differentiable rewards (requires the reward model to be a neural network the model can "see through").
*   **DPO:** A non-RL preference optimization method.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
DanceGRPO significantly outperforms all baselines. In **Stable Diffusion**, it raised the HPS score by over $50\%$. Most impressively, in **Video Generation**, it achieved a **$181\%$ increase in Motion Quality** scores. This suggests the model learned to create much smoother and more realistic physical movements through RL.

## 6.2. Data Presentation (Tables)

The following are the results from Table 2 of the original paper (Stable Diffusion v1.4):

<table>
<thead>
<tr>
<th>Models</th>
<th>HPS-v2.1</th>
<th>CLIP Score</th>
<th>Pick-a-Pic</th>
<th>GenEval</th>
</tr>
</thead>
<tbody>
<tr>
<td>Stable Diffusion (Base)</td>
<td>0.239</td>
<td>0.363</td>
<td>0.202</td>
<td>0.421</td>
</tr>
<tr>
<td>Stable Diffusion w/ HPS-v2.1</td>
<td>0.365</td>
<td>0.380</td>
<td>0.217</td>
<td>0.521</td>
</tr>
<tr>
<td>**Stable Diffusion w/ HPS & CLIP**</td>
<td>**0.335**</td>
<td>**0.395**</td>
<td>**0.215**</td>
<td>**0.522**</td>
</tr>
</tbody>
</table>

The following are the results from Table 5 of the original paper (HunyuanVideo):

<table>
<thead>
<tr>
<th>Benchmarks</th>
<th>VQ (Visual Quality)</th>
<th>MQ (Motion Quality)</th>
<th>TA (Text Alignment)</th>
<th>VisionReward</th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline</td>
<td>4.51</td>
<td>1.37</td>
<td>1.75</td>
<td>0.124</td>
</tr>
<tr>
<td>**Ours (DanceGRPO)**</td>
<td>**7.03 (+56%)**</td>
<td>**3.85 (+181%)**</td>
<td>**1.59**</td>
<td>**0.128**</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies
*   **Timestep Selection:** The authors found that training on the **initial $30\%$ of noise** is crucial for foundational quality, but skipping the late-stage refinement steps entirely leads to "oily" or unnatural textures.
*   **Binary Rewards:** They tested if the model could learn from simple "Yes/No" (1 or 0) rewards. Surprisingly, DanceGRPO successfully learned the distribution even with this sparse, non-continuous feedback.

    The following figure (Figure 10 from the original paper) visualizes the impact of different reward models:

    ![该图像是三张杯子饮品的对比图，左侧为原始图像，中间为使用 HPS 分数优化后的图像，右侧为同时使用 HPS 分数和 CLIP 分数优化后的图像。图像展示了不同优化方法对饮料视觉效果的提升。](images/10.jpg)
    *Figure 10: Using only HPS (middle) can lead to unnatural "oily" textures. Combining HPS with CLIP (right) preserves natural details while improving aesthetics.*

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
DanceGRPO is a milestone in visual alignment. It successfully migrates the stability of **Group Relative Policy Optimization** from the world of Large Language Models to **Visual Generative AI**. By reformulating sampling as SDEs and using group-based advantages, the authors have created a framework that is stable, scalable, and memory-efficient.

## 7.2. Limitations & Future Work
*   **Inference Speed:** While training is more stable, the use of SDEs (stochastic sampling) during training can be slower than pure deterministic ODEs.
*   **Reward Model Dependency:** The model is only as good as its judge. If the reward model has biases (like the "oily" texture bias in HPS), the generative model will inherit them.
*   **Future Work:** The authors suggest exploring more complex "rule-based" rewards (like using Multimodal LLMs as judges) and extending this to unified multimodal models that generate text and images simultaneously.

## 7.3. Personal Insights & Critique
**Insights:**
The most brilliant move in this paper is the **Shared Initialization Noise**. In previous RL attempts, if a model generated a bad image, the model didn't know if it was because the model was "dumb" or because the random starting noise was just "unlucky." By forcing the group to start with the same noise, the reward signals become much cleaner, allowing the model to learn much faster.

**Critique:**
While the results are impressive, the paper uses 32 to 64 **H800 GPUs** for training. This level of compute is out of reach for most academic researchers, making the "scalability" a feature mostly for large industrial labs. Additionally, the drop in Text-Alignment (TA) in the video results (Table 5) suggests that optimizing for motion might sometimes come at a slight cost to sticking strictly to the prompt, a trade-off that needs more investigation.