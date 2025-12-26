# 1. Bibliographic Information

## 1.1. Title
Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

## 1.2. Authors
Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, Robin Rombach. All authors are affiliated with **Stability AI**.

## 1.3. Journal/Conference
This paper was released as a technical report/preprint on arXiv (2024). Given the authors' history (creators of **Stable Diffusion**), this work serves as the foundational technical document for **Stable Diffusion 3 (SD3)**. ArXiv is the primary venue for rapid dissemination of state-of-the-art research in generative AI.

## 1.4. Publication Year
2024 (Published on March 5, 2024).

## 1.5. Abstract
The paper investigates the use of **rectified flows**—a generative model formulation that connects data and noise in a straight line—for high-resolution image synthesis. The authors improve existing noise sampling techniques by biasing them toward perceptually relevant scales. They introduce a novel **multimodal transformer (MM-DiT)** architecture that uses separate weight streams for image and text tokens, enabling bidirectional information flow. The study demonstrates predictable scaling trends (scaling laws) and shows that their 8B parameter model outperforms existing state-of-the-art models like **DALL-E 3** and **SDXL** in typography, prompt following, and visual quality.

## 1.6. Original Source Link
*   **Original Source Link:** [https://arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)
*   **PDF Link:** [https://arxiv.org/pdf/2403.03206v1.pdf](https://arxiv.org/pdf/2403.03206v1.pdf)
*   **Status:** Preprint (Technical Report).

# 2. Executive Summary

## 2.1. Background & Motivation
Generative modeling, specifically text-to-image synthesis, has been dominated by **diffusion models**. While effective, these models often rely on complex, curved paths to transform random noise into structured images, which requires many computational steps during inference (sampling). 

The core problems the paper addresses are:
1.  **Sampling Efficiency:** Traditional diffusion paths are curved; straight paths (**rectified flows**) are theoretically easier to simulate but hadn't been proven at the scale of massive text-to-image models.
2.  **Architecture Limitations:** Existing backbones (like **UNets** or standard **Transformers**) often treat text conditioning as a secondary input rather than an equal modality, limiting the model's ability to handle complex spatial reasoning and typography.
3.  **Predictability:** The field lacked clear "scaling laws" for diffusion models similar to those found in Large Language Models (LLMs).

    The paper's entry point is the systematic marriage of **Rectified Flow** theory with a **Multimodal Transformer** architecture, specifically optimized for scaling to billions of parameters.

## 2.2. Main Contributions / Findings
*   **Improved Rectified Flow Training:** Introduced new noise sampling schedules (like `Logit-Normal` sampling) that prioritize "middle" timesteps, which are more critical for learning image structure.
*   **MM-DiT Architecture:** A transformer-based backbone with separate weights for text and image tokens, allowing the text to "evolve" alongside the image during the generation process.
*   **Scaling Laws for Generative Media:** Demonstrated that validation loss in image and video synthesis follows a predictable power-law relationship with compute and model size.
*   **State-of-the-Art Performance:** The 8B parameter model achieves superior human preference ratings over **DALL-E 3** and **Midjourney v6**, particularly in typography and prompt adherence.
*   **Resolution Shifting:** A novel method to adjust the noise schedule based on image resolution, ensuring that the model doesn't get "confused" by the change in pixel count.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, a novice needs to grasp the following:

*   **Generative Modeling:** The task of creating new data (images) that look like they belong to a training set.
*   **Diffusion Models:** These models work by taking an image, gradually adding noise until it is unrecognizable, and then learning to reverse that process.
*   **Latent Space:** Instead of working on raw pixels (which is computationally heavy), models often work in a "compressed" mathematical space called a `latent space`, created by an `Autoencoder`.
*   **Ordinary Differential Equation (ODE):** In this context, an `ODE` is a mathematical way to describe the continuous path from random noise to a clean image. If the path is a straight line, it is a `Rectified Flow`.
*   **Transformer Architecture:** A neural network design that uses `Self-Attention` to weight the importance of different parts of the input data (e.g., how a word in a prompt relates to a specific area in an image).

## 3.2. Previous Works
The authors build upon several pillars:
*   **Stable Diffusion / LDM (Rombach et al., 2022):** Established the standard for `Latent Diffusion Models`. The core formula for the `Attention` mechanism used in these transformers is:
    \$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    \$
    where $Q$ (Query) represents what we are looking for, $K$ (Key) is what is available, $V$ (Value) is the content, and $d_k$ is the dimension of the keys.
*   **DiT (Diffusion Transformer) (Peebles & Xie, 2023):** Replaced the standard `UNet` (a convolutional architecture) with a `Transformer`, proving that transformers can scale better for images.
*   **Rectified Flow (Liu et al., 2022):** Introduced the idea of "straightening" the path between noise and data to make sampling faster.

## 3.3. Technological Evolution
Early models used **GANs** (Generative Adversarial Networks), which were unstable to train. **Diffusion Models** (like DDPM) brought stability but were slow. **Latent Diffusion** (Stable Diffusion 1.5/2.1/XL) moved the process to a compressed space. This paper represents the next step: moving from a `UNet` to a `Multimodal Transformer` and from standard `Diffusion` to `Rectified Flows`.

## 3.4. Differentiation Analysis
Unlike previous models that use **Cross-Attention** (where text is a fixed "context" for the image), this paper’s **MM-DiT** allows image tokens to influence text tokens and vice versa. Furthermore, while others use fixed noise schedules, this paper introduces `Logit-Normal` sampling to make training more efficient.

# 4. Methodology

## 4.1. Principles
The core intuition is that a **Straight Path** is the most efficient way to travel from noise to an image. If the model learns to follow a straight line, it can reach the destination (a clean image) in very few steps without "getting lost" or accumulating errors.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Simulation-Free Training of Flows
The model defines the mapping between noise $x_1$ and data $x_0$ using an **Ordinary Differential Equation (ODE)**:
\$
dy_t = v_{\Theta}(y_t, t) dt
\$
where:
*   $y_t$: The state of the data at time $t$.
*   $v_{\Theta}$: The "velocity" or the vector field predicted by the neural network with weights $\Theta$.
*   `dt`: An infinitesimal change in time.

    To train this, the authors define a **probability path** $p_t$ between data and noise:
\$
z_t = a_t x_0 + b_t \epsilon \quad \text{where} \ \epsilon \sim \mathcal{N}(0, I)
\$
In this formula:
*   $z_t$: The noisy sample at time $t$.
*   $a_t$: A function defining how much of the original data $x_0$ remains.
*   $b_t$: A function defining how much noise $\epsilon$ is added.
*   $\mathcal{N}(0, I)$: A standard normal distribution (random noise).

### 4.2.2. The Vector Field and Objective Function
The goal is to regress the "conditional vector field" $u_t$. By differentiating $z_t$ with respect to $t$, the authors derive the target velocity (Equation 9):
\$
u_t(z_t | \epsilon) = \frac{a_t'}{a_t} z_t - \frac{b_t}{2} \lambda_t' \epsilon
\$
where $\lambda_t = \log \frac{a_t^2}{b_t^2}$ is the **log signal-to-noise ratio (SNR)**.

The network is trained using the **Conditional Flow Matching (CFM)** loss (Equation 12):
\$
\mathcal{L}_w(x_0) = - \frac{1}{2} \mathbb{E}_{t \sim \mathcal{U}(t), \epsilon \sim \mathcal{N}(0, I)} \left[ w_t \lambda_t' \| \epsilon_{\Theta}(z_t, t) - \epsilon \|^2 \right]
\$
Symbols:
*   $w_t$: A time-dependent weighting factor.
*   $\epsilon_{\Theta}$: The noise predicted by the network.
*   $\lambda_t'$: The derivative of the log SNR.

### 4.2.3. Rectified Flow Specifics
For **Rectified Flow (RF)**, the authors choose a linear interpolation (Equation 13):
\$
z_t = (1 - t) x_0 + t \epsilon
\$
This makes the path a **straight line**. Here, $a_t = 1-t$ and $b_t = t$.

### 4.2.4. Novel Noise Sampling: Logit-Normal
The authors argue that the "middle" of the generation process (where $t \approx 0.5$) is the hardest and most important part to learn. They use a **Logit-Normal distribution** to sample $t$ more frequently in the middle (Equation 19):
\$
\pi_{ln}(t; m, s) = \frac{1}{s \sqrt{2 \pi}} \frac{1}{t (1 - t)} \exp \left( - \frac{(\mathrm{logit}(t) - m)^2}{2 s^2} \right)
\$
where:
*   $m$: Location parameter (biases toward noise or data).
*   $s$: Scale parameter (how "wide" the bias is).

### 4.2.5. MM-DiT Architecture
The system architecture is a **Multimodal Diffusion Transformer**. The following figure (Figure 2 from the original paper) shows the system architecture:

![Figure 2. Our model architecture. Concatenation is indicated by $\\odot$ and element-wise multiplication by $^ *$ The RMS-Norm for $Q$ and $K$ can be added to stabilize training runs. Best viewed zoomed in.](images/2.jpg)

**Step-by-Step Architecture Logic:**
1.  **Input Patching:** The latent image is flattened into $2 \times 2$ patches.
2.  **Dual Streams:** Unlike standard transformers, MM-DiT maintains separate streams for **Image Tokens** and **Text Tokens**.
3.  **Bidirectional Attention:** In the attention block, the sequences are joined. Image tokens "attend" to text tokens to understand the prompt, while text tokens "attend" to image tokens to refine their representation based on the visual context.
4.  **Modulation:** The timestep $t$ and pooled text embeddings condition the layers via a modulation mechanism (scaling and shifting the features).
5.  **QK-Normalization:** To prevent training instabilities in large models, they normalize the `Query` and `Key` vectors using `RMSNorm`:
    \$
    \mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n} \sum x_i^2}}
    \$

### 4.2.6. Resolution Shifting
When moving to higher resolutions (e.g., $1024 \times 1024$), there are more pixels, which effectively changes the signal-to-noise ratio. They shift the timesteps using Equation 23:
\$
t_m = \frac{\sqrt{\frac{m}{n}} t_n}{1 + (\sqrt{\frac{m}{n}} - 1) t_n}
\$
where $n$ is the original pixel count and $m$ is the new pixel count. This ensures the model "sees" the same level of noise regardless of resolution.

# 5. Experimental Setup

## 5.1. Datasets
*   **ImageNet:** A standard dataset of 1.2M images with 1,000 classes. Used for early formulation testing.
*   **CC12M (Conceptual 12M):** A dataset of 12M web images with captions. Used for text-to-image architecture testing.
*   **Internal Large-Scale Dataset:** For final scaling (8B model), they used a massive dataset filtered for aesthetics and safety.
*   **Synthetic Captions:** They used **CogVLM** to re-caption images, providing more detail than original web alt-text.

## 5.2. Evaluation Metrics

### 5.2.1. FID (Fréchet Inception Distance)
1.  **Conceptual Definition:** Quantifies the similarity between generated images and real images by comparing statistics of features extracted from a pre-trained deep network. Lower is better.
2.  **Mathematical Formula:**
    \$
    d^2 = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
    \$
3.  **Symbol Explanation:**
    *   $\mu_r, \Sigma_r$: Mean and covariance of real image features.
    *   $\mu_g, \Sigma_g$: Mean and covariance of generated image features.
    *   $\mathrm{Tr}$: The trace operator (sum of diagonal elements).

### 5.2.2. CLIP Score
1.  **Conceptual Definition:** Measures how well the generated image matches the text prompt by calculating the cosine similarity between their embeddings. Higher is better.
2.  **Mathematical Formula:**
    \$
    \text{Score} = w \cdot \cos(\theta_{image}, \theta_{text})
    \$
3.  **Symbol Explanation:**
    *   $\theta_{image}$: Vector embedding of the image.
    *   $\theta_{text}$: Vector embedding of the text.
    *   $w$: A scaling factor (usually 100).

### 5.2.3. GenEval
A benchmark specifically designed to test **Prompt Adherence**, checking if the model correctly places objects, counts them, and attributes colors.

## 5.3. Baselines
*   **SDXL:** The previous state-of-the-art open-source model.
*   **DALL-E 3:** OpenAI's closed-source model known for prompt following.
*   **PixArt-α:** A competitive transformer-based diffusion model.

# 6. Results & Analysis

## 6.1. Core Results Analysis
*   **Rectified Flow Efficiency:** As seen in Figure 3, Rectified Flows maintain high quality (low FID) even when sampling steps are reduced (as low as 10-25 steps), whereas standard diffusion models degrade rapidly.
*   **MM-DiT vs. Others:** In Figure 4, MM-DiT consistently achieves lower validation loss and higher CLIP scores compared to `UViT` and `Cross-Attention DiT`.
*   **Scaling Laws:** The authors show that doubling compute (FLOPs) leads to a predictable decrease in validation loss, following a linear trend on a log-log scale.

## 6.2. Data Presentation (Tables)

The following are the results from Table 1 of the original paper, showing the ranking of different noise formulations:

<table>
<thead>
<tr>
<th rowspan="2">Variant</th>
<th colspan="3">Rank Averaged Over</th>
</tr>
<tr>
<th>All</th>
<th>5 Steps</th>
<th>50 Steps</th>
</tr>
</thead>
<tbody>
<tr>
<td>rf/lognorm(0.00, 1.00)</td>
<td>1.54</td>
<td>1.25</td>
<td>1.50</td>
</tr>
<tr>
<td>rf/lognorm(1.00, 0.60)</td>
<td>2.08</td>
<td>3.50</td>
<td>2.00</td>
</tr>
<tr>
<td>eps/linear (Standard LDM)</td>
<td>2.88</td>
<td>4.25</td>
<td>2.75</td>
</tr>
<tr>
<td>rf (Standard Rectified Flow)</td>
<td>5.67</td>
<td>6.50</td>
<td>5.75</td>
</tr>
</tbody>
</table>

The following are the results from Table 5, comparing the 8B model (depth 38) against other SOTA models on the **GenEval** benchmark:

<table>
<thead>
<tr>
<th>Model</th>
<th>Overall Score</th>
<th>Single Object</th>
<th>Two Objects</th>
<th>Counting</th>
<th>Color Attribution</th>
</tr>
</thead>
<tbody>
<tr>
<td>SDXL</td>
<td>0.55</td>
<td>0.98</td>
<td>0.74</td>
<td>0.39</td>
<td>0.23</td>
</tr>
<tr>
<td>DALL-E 3</td>
<td>0.67</td>
<td>0.96</td>
<td>0.87</td>
<td>0.47</td>
<td>0.45</td>
</tr>
<tr>
<td>Ours (depth=38)</td>
<td>0.68</td>
<td>0.98</td>
<td>0.84</td>
<td>0.66</td>
<td>0.43</td>
</tr>
<tr>
<td>Ours (depth=38) + DPO</td>
<td>0.74</td>
<td>0.99</td>
<td>0.94</td>
<td>0.72</td>
<td>0.60</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies
*   **T5 Text Encoder:** Removing the `T5-XXL` encoder results in a loss of typography and complex prompt understanding, but aesthetic quality remains high.
*   **QK-Normalization:** Without it, training the 8B model becomes unstable (NaNs occur) at high resolutions. Figure 5 illustrates the "attention logit growth" problem that this normalization fixes.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper proves that **Rectified Flow Transformers** are the current frontier for high-resolution image synthesis. By combining a "straight path" noise schedule with a multimodal transformer architecture, the authors created **Stable Diffusion 3**, which is not only more efficient but also more capable of following complex prompts and generating accurate text/typography than its predecessors.

## 7.2. Limitations & Future Work
*   **Computational Cost:** Training an 8B model requires massive GPU resources.
*   **T5 VRAM:** The `T5` encoder alone takes ~19GB of VRAM, making inference difficult on consumer hardware without quantization.
*   **Future Directions:** Scaling the same principles to high-resolution video (preliminary results shown in Section 5.3.3) and further refining the "Direct Preference Optimization" (DPO) for image aesthetics.

## 7.3. Personal Insights & Critique
This paper is a masterclass in **engineering at scale**. While the individual components (Transformers, Rectified Flow, Logit-Normal) existed in isolation, their integration is what makes this work significant. 
*   **Inspiration:** The "Resolution Shifting" logic is a clever way to handle variable aspect ratios and resolutions, which was a major pain point in Stable Diffusion 1.5.
*   **Critique:** The paper mentions video synthesis but provides significantly less detail on the video architecture compared to the image model. Additionally, the reliance on a "50/50 mix" of synthetic captions suggests that the quality of human data is currently a bottleneck for the industry.
*   **Application:** The MM-DiT architecture could likely be applied to other multimodal tasks, such as generating audio from video or complex 3D scenes from text.