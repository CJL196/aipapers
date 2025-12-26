# 1. Bibliographic Information

## 1.1. Title
UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation

The title clearly states the paper's core contributions:
- **UnityVideo:** The name of the proposed framework.
- **Unified Multi-Modal Multi-Task Learning:** The core methodology, which involves training a single model on various data types (modalities) and for different objectives (tasks) simultaneously.
- **Enhancing World-Aware Video Generation:** The primary goal, which is to improve the model's ability to generate videos that are consistent with real-world physics, geometry, and motion.

## 1.2. Authors
Jiehui Huang, Yuechen Zhang, Xu He, Yuan Gao, Zhi Cen, Bin Xia, Yan Zhou, Xin Tao, Pengfei Wan, and Jiaya Jia.

The authors are affiliated with several prominent institutions:
- Hong Kong University of Science and Technology (HKUST)
- The Chinese University of Hong Kong (CUHK)
- Tsinghua University
- Kling Team, Kuaishou Technology

  This collaboration brings together top academic researchers (including Prof. Jiaya Jia, a highly respected figure in computer vision) and a leading industry team from Kuaishou, a major video-sharing platform. This suggests that the research is not only academically rigorous but also grounded in practical, large-scale application needs.

## 1.3. Journal/Conference
The paper was submitted to arXiv with a publication date of December 8, 2025. This indicates it is a preprint, likely intended for submission to a top-tier computer vision or machine learning conference such as CVPR, ICCV, NeurIPS, or ICLR for the 2025 or 2026 cycle. These conferences are highly competitive and are the premier venues for publishing cutting-edge research in the field.

## 1.4. Publication Year
2025 (as per the metadata on arXiv).

## 1.5. Abstract
The abstract summarizes that recent video generation models, while impressive, are limited by their reliance on a single modality (typically RGB video). This constrains their understanding of the world. To overcome this, the paper introduces **UnityVideo**, a unified framework that jointly learns from multiple modalities (segmentation masks, skeletons, DensePose, optical flow, depth maps) and across different training tasks. The framework's core innovations are **(1) dynamic noising** to unify different training objectives and **(2) a modality switcher with an in-context learner** to handle diverse data types. The authors also contribute a large-scale dataset (`OpenUni`) with 1.3 million samples. The key findings are that this unified approach accelerates model convergence, improves zero-shot generalization, and produces videos with superior quality, consistency, and alignment with physical world principles.

## 1.6. Original Source Link
- **Original Source:** https://arxiv.org/abs/2512.07831
- **PDF Link:** https://arxiv.org/pdf/2512.07831v1.pdf
- **Publication Status:** This is a preprint available on arXiv and has not yet undergone formal peer review for an official publication.

# 2. Executive Summary

## 2.1. Background & Motivation
- **Core Problem:** State-of-the-art video generation models are primarily trained on massive amounts of RGB video data. While this has led to visually stunning results, these models often lack a deep, holistic understanding of the physical world. They may generate videos with incorrect physics, inconsistent object geometry, or unnatural motion because they are essentially learning statistical patterns from pixels rather than underlying world principles.

- **Existing Gaps:** The authors draw an analogy to Large Language Models (LLMs). LLMs achieved powerful reasoning and generalization by unifying diverse text-based "sub-modalities" like natural language, code, and mathematical expressions into a single training paradigm. In contrast, video models have focused on scaling up with just one modality (RGB video), which is like training an LLM only on plain text. Previous attempts to incorporate other visual modalities (like depth or skeletons) have been limited, often focusing on a single extra modality or a one-way interaction (e.g., using a depth map to control video generation, but not learning from it to improve the model's intrinsic understanding of geometry).

- **Innovative Idea:** The paper's central hypothesis is that **unifying multiple visual sub-modalities and related tasks within a single framework** can create a synergistic learning effect. By forcing a single model to concurrently generate RGB video, estimate depth, predict optical flow, understand human poses, and perform segmentation, it can develop a more comprehensive and robust internal representation of the "world." This shared knowledge should then benefit all tasks, leading to better generalization and more physically plausible video generation.

## 2.2. Main Contributions / Findings
The paper presents three main contributions:

1.  **The UnityVideo Framework:** A novel, unified architecture designed for multi-modal and multi-task learning in video. Its key technical innovations are:
    - A **dynamic noise scheduling** strategy that allows the model to seamlessly switch between three distinct training tasks (conditional generation, modality estimation, and joint generation) within a single training cycle.
    - A **modality-adaptive switcher** and an **in-context learner**, which are lightweight mechanisms that enable the model to process and differentiate between multiple heterogeneous modalities (depth, flow, skeleton, etc.) using shared parameters, promoting efficient knowledge transfer.

2.  **A Large-Scale Dataset and Benchmark:**
    - **OpenUni:** A new, large-scale dataset containing 1.3 million video clips, each paired with annotations for five auxiliary modalities: depth, optical flow, DensePose, skeleton, and segmentation. This resource is crucial for enabling research into unified video modeling.
    - **UniBench:** A new evaluation benchmark designed for unified video models. It includes high-quality synthetic videos from Unreal Engine with perfect ground-truth data for tasks like depth estimation, ensuring fair and accurate evaluation.

3.  **Demonstrated Superior Performance:** The experimental results show that `UnityVideo`:
    - **Accelerates convergence** compared to training on single modalities.
    - Achieves **state-of-the-art performance** across a range of tasks, outperforming specialized models in video generation, depth estimation, and video segmentation.
    - Exhibits **strong zero-shot generalization**, for example, by applying segmentation learned on "two persons" to "two objects" or by estimating skeletons for animals despite being trained primarily on humans. This suggests the model is learning underlying concepts rather than just memorizing patterns.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
- **Diffusion Models:** A class of generative models that learn to create data by reversing a gradual noising process. The process involves two steps:
    1.  **Forward Process:** Starting with a real data sample (e.g., an image), a small amount of Gaussian noise is added iteratively over many steps until the data becomes pure noise.
    2.  **Reverse Process:** A neural network is trained to reverse this process. Given a noised sample and the corresponding timestep, the model learns to predict the noise that was added, allowing it to gradually "denoise" a random input back into a coherent data sample.

- **Flow Matching:** A more recent and often more efficient alternative to diffusion models. Instead of a stochastic noising/denoising process, flow matching frames generation as learning a continuous vector field (or "flow") that transports samples from a simple prior distribution (like Gaussian noise) to the target data distribution along a deterministic path. This is defined by an Ordinary Differential Equation (ODE). The model is trained to predict the "velocity" vector at any point along this path, which tells the sample how to move to get closer to a real data sample. `UnityVideo` uses this framework.

- **Transformer & DiT (Diffusion Transformer):**
    - The **Transformer** is a neural network architecture originally designed for natural language processing. Its key component is the **self-attention mechanism**, which allows the model to weigh the importance of different parts of the input sequence when processing any given part. This enables it to capture long-range dependencies effectively.
    - A **Diffusion Transformer (DiT)** adapts the Transformer architecture for use in diffusion models. It treats an image or video as a sequence of "tokens" (similar to words in a sentence). For example, an image can be broken into a grid of patches, and each patch is flattened into a vector token. These tokens are then fed into a Transformer, which learns to perform the denoising task. DiTs have shown excellent scalability and performance.

- **Multi-Modality:** In machine learning, a "modality" refers to a specific type or format of data. In this paper, the modalities are:
    - **RGB Video:** Standard color video.
    - **Depth Maps:** A grayscale video where each pixel's intensity represents its distance from the camera. It encodes 3D geometry.
    - **Optical Flow:** Represents the motion of objects between consecutive frames. It's often visualized as a 2D vector field.
    - **Segmentation Masks:** An image where pixels belonging to a specific object or category are colored the same, effectively outlining objects.
    - **Human Skeletons:** A set of keypoints (joints) representing a person's pose and movement.
    - **DensePose:** A more detailed representation than a skeleton, mapping all pixels on a human body to a 3D surface model.

## 3.2. Previous Works
The authors situate their work in the context of two main research directions:

1.  **Controllable Video Generation:** Many models use auxiliary modalities as control signals. For instance, a user provides a sequence of skeleton poses, and the model generates a video of a person moving accordingly. Examples include `VACE` and `Full-DiT`. However, this is a one-way interaction; the model learns to follow the control but doesn't necessarily improve its intrinsic understanding of human motion.

2.  **Video Reconstruction/Estimation:** Other models focus on the inverse problem: predicting an auxiliary modality from an RGB video. For example, `DepthCrafter` and `Geo4D` estimate depth maps from a video. This helps the model "understand" geometry, but the goal is estimation, not generation.

    A few recent works have explored **bidirectional interactions**, where the model learns relationships between RGB and another modality in both directions. For example, `Aether` and `GeoVideo` jointly model video and geometric data (like depth), showing that this shared representation leads to mutual benefits. `EgoTwin` does something similar for skeletons and video.

However, the authors point out that these works are still limited. They typically only unify *two* modalities (e.g., RGB + depth) or focus on a single architectural approach. No prior work had attempted to build a single, unified framework that jointly learns from a wide diversity of modalities and tasks simultaneously.

## 3.3. Technological Evolution
The field has progressed as follows:
1.  **Unconditional Video Generation:** Early models generated short, often low-quality videos from random noise.
2.  **Text-to-Video Generation:** Models like Sora and Kling scaled up training to generate high-fidelity videos from text prompts. These models still primarily use RGB video data.
3.  **Controllable Video Generation:** Researchers added the ability to control generation using conditions like depth maps, skeletons, or edge maps. This improved user control but often treated the control signal as an external input.
4.  **Bidirectional and Joint Modeling:** A few models started to jointly learn representations for RGB and one other modality, showing that this improves understanding of concepts like geometry or motion.
5.  **Unified Multi-Modal, Multi-Task Learning (This Paper):** `UnityVideo` represents the next logical step, proposing a highly general framework to unify many modalities and tasks, aiming for a more holistic "world model" that benefits from the synergy of diverse data and objectives.

## 3.4. Differentiation Analysis
The core innovation of `UnityVideo` compared to previous works is its **scope and unification**.
- **Multi-Modal:** Instead of just RGB + 1 modality, it integrates *five* auxiliary modalities, covering geometry (depth), motion (optical flow), and human-centric understanding (skeleton, DensePose, segmentation).
- **Multi-Task:** It doesn't just do conditional generation or estimation. It trains on three different tasks—conditional generation, estimation, and joint generation—*simultaneously* in a single model, using a single set of weights.
- **Unified Framework:** The proposed architectural components (`dynamic noising`, `modality switcher`, `in-context learner`) are specifically designed to manage this complexity and enable positive knowledge transfer, whereas previous works used more siloed approaches.

# 4. Methodology
The core principle of `UnityVideo` is to create a single, versatile model that can learn from and generate multiple visual modalities by unifying different training tasks within a shared architecture. This is achieved through three key technical components.

The overall architecture is a Diffusion Transformer (`DiT`) that processes tokens from RGB video ($V_r$), an auxiliary modality ($V_m$), and text prompts ($C$).

![该图像是示意图，展示了UnityVideo框架的多模态视频生成结构。图中展示了动态噪声、模态切换器、以及不同的生成策略，包括条件生成、视频估计和联合生成等方法。主要结构使用了自注意力机制和上下文交叉注意力，通过这些设计来增强世界意识的视频生成能力。](images/3.jpg)
*该图像是示意图，展示了UnityVideo框架的多模态视频生成结构。图中展示了动态噪声、模态切换器、以及不同的生成策略，包括条件生成、视频估计和联合生成等方法。主要结构使用了自注意力机制和上下文交叉注意力，通过这些设计来增强世界意识的视频生成能力。*

## 4.1. Unifying Multiple Tasks via Dynamic Noising
`UnityVideo` is trained to perform three distinct tasks, which are dynamically selected during each training iteration. This prevents the model from specializing in one task and forgetting others. The tasks are unified within the flow matching framework.

1.  **Conditional Generation ($u(V_r | V_m, C)$):** Generate an RGB video ($V_r$) conditioned on a clean auxiliary modality ($V_m$) and a text prompt ($C$). In this task, the RGB video tokens are corrupted with noise, and the model learns to denoise them using the clean modality tokens as guidance.
2.  **Modality Estimation ($u(V_m | V_r)$):** Estimate an auxiliary modality ($V_m$) from a clean RGB video ($V_r$). Here, the modality tokens are noised, and the model learns to reconstruct them using the clean RGB video tokens as input.
3.  **Joint Generation ($u(V_r, V_m | C)$):** Generate both the RGB video and the auxiliary modality simultaneously from pure noise, conditioned only on a text prompt ($C$). In this case, both the RGB and modality tokens are corrupted with noise, and the model learns to denoise them together.

    **Dynamic Task Routing:** At each training step, one of these three tasks is randomly sampled with probabilities $p_{cond}$, $p_{est}$, and $p_{joint}$. The probabilities are set inversely proportional to task difficulty ($p_{cond} < p_{est} < p_{joint}$) to balance learning. This dynamic switching allows the model to receive gradients for all three objectives concurrently, fostering synergy and preventing catastrophic forgetting.

## 4.2. Unifying Multiple Modalities via Adaptive Learning
To handle the diverse nature of the five auxiliary modalities within a single model, `UnityVideo` introduces two lightweight components.

### 4.2.1. In-Context Learner
This component provides **semantic-level** differentiation between modalities. Instead of just using a content-describing text prompt ($C_r$, e.g., "a person walking"), the model also receives a modality-specific prompt ($C_m$, e.g., "depth map" or "human skeleton").

The model uses a dual-branch cross-attention mechanism:
- One branch computes cross-attention between the RGB video tokens ($V_r$) and the content prompt ($V_r' = \text{CrossAttn}(V_r, C_r)$).
- The other branch computes cross-attention between the auxiliary modality tokens ($V_m$) and the modality type prompt ($V_m' = \text{CrossAttn}(V_m, C_m)$).

  This simple design allows the model to use its contextual reasoning ability to understand what *type* of data it is processing. This is shown to be powerful for zero-shot generalization. For example, by learning the concept of "segmentation" from the prompt "segment two persons," the model can later apply it to a new task like "segment two objects."

### 4.2.2. Modality-Adaptive Switcher
This component provides **architecture-level** differentiation. It extends the `AdaLN-Zero` (Adaptive Layer Normalization) mechanism commonly used in `DiT`s. In a standard `DiT`, `AdaLN` generates modulation parameters (scale $\gamma$, shift $\beta$, and gate $\alpha$) based on the timestep embedding to condition the network.

`UnityVideo` enhances this by making the parameters modality-specific. It maintains a learnable embedding list $\mathbf{L}_m = \{L_1, L_2, ..., L_k\}$ for the $k$ different modalities. The modulation parameters are then computed as:
\$
\gamma_m, \beta_m, \alpha_m = \mathbf{MLP}(L_m + t_{emb})
\$
where $L_m$ is the embedding for the current modality and $t_{emb}$ is the timestep embedding. This allows each `DiT` block to adjust its behavior specifically for the modality it is processing, acting like a "switch." This design is modular and enables plug-and-play selection of modalities at inference time.

## 4.3. Training Strategy and Objective
### 4.3.1. Curriculum Learning
Training a model on all five modalities from scratch is difficult. The paper proposes a two-stage curriculum:
- **Stage 1:** The model is first trained only on **pixel-aligned** modalities (optical flow, depth, DensePose) using a human-centric dataset. These modalities have a direct pixel-to-pixel correspondence with the RGB frame, making it easier for the model to learn spatial relationships.
- **Stage 2:** Training is expanded to include all five modalities, including **pixel-unaligned** ones (segmentation, skeleton), and a more diverse dataset covering general scenes.

  This progressive strategy helps the model build a strong foundation before tackling more complex and abstract modalities.

### 4.3.2. Training Objective
The model is trained using the Conditional Flow Matching objective. Depending on the task sampled in a given step, one of the following three losses is computed. In these equations, $u_\theta$ is the model, $r$ and $m$ represent RGB and modality latents, $r_0, m_0$ are clean latents, $r_1, m_1$ are noise, and $v_r, v_m$ are the velocity fields to be predicted.

1.  **Conditional Generation Loss ($\mathcal{L}_{cond}$):** The model predicts the velocity field for the RGB video, given the noised RGB latent $r_t$ and the clean modality latent $m_0$.
    \$
    \mathcal{L}_{\mathrm{cond}}(\theta; t) = \mathbb{E}\left[\Vert u_{\theta}(r_t, [m_0, c_{\mathrm{txt}}], t) - v_r \Vert^2\right]
    \$

2.  **Estimation Loss ($\mathcal{L}_{est}$):** The model predicts the velocity field for the auxiliary modality, given the noised modality latent $m_t$ and the clean RGB latent $r_0$.
    \$
    \mathcal{L}_{\mathrm{est}}(\theta; t) = \mathbb{E}\left[\Vert u_{\theta}(m_t, r_0, t) - v_m \Vert^2\right]
    \$

3.  **Joint Generation Loss ($\mathcal{L}_{joint}$):** The model predicts the velocity fields for both the RGB video and the modality, given both noised latents $r_t$ and $m_t$.
    \$
    \mathcal{L}_{\mathrm{joint}}(\theta; t) = \mathbb{E}\left[\Vert u_{\theta}([r_t, m_t], c_{\mathrm{txt}}, t) - [v_r, v_m] \Vert^2\right]
    \$

By randomly choosing one of these losses at each step, the model is optimized for all three tasks within a single, unified training process.

# 5. Experimental Setup

## 5.1. Datasets
- **Training Dataset (`OpenUni`):** A large-scale dataset of 1.3 million video clips created by the authors. It was compiled from multiple sources, including `Koala36M` and `OpenS2V`. For each video, five corresponding modalities (optical flow, depth, DensePose, skeleton, segmentation) were extracted using pre-trained models. The dataset is balanced to prevent overfitting to a specific source or modality.

  ![Figure 4. OpenUni dataset. OpenUni contains 1.3M pairs of unified multimodal data, designed to enrich video modalities with more comprehensive world perception.](images/4.jpg)
  *该图像是示意图，展示了UnityVideo框架在数据收集、视频过滤和多模态提取过程中的各个步骤。图中提供了一个包含130万对统一多模态数据的数据集，旨在增强视频生成的世界感知能力。*

- **Evaluation Benchmarks:**
    - **`VBench`:** A well-established public benchmark for evaluating video generative models on aspects like video quality, consistency, and motion.
    - **`UniBench`:** A new benchmark created by the authors for evaluating unified models. It has two components:
        1.  200 synthetic video samples created in **Unreal Engine (UE)**. This provides perfect, noise-free ground-truth depth and optical flow data, which is crucial for quantitatively evaluating estimation tasks.
        2.  200 curated real-world video samples with annotations for all modalities, used to assess controllable generation and segmentation.

            ![该图像是一个示意图，展示了多模态视频生成模型的不同输入数据，包括 RGB 图像、光流和深度图。各个部分显示了用于评估 Text2Video 和 Control Generation 任务的真实数据，有助于理解 UnityVideo 的方法与效果。](images/10.jpg)
            *该图像是一个示意图，展示了多模态视频生成模型的不同输入数据，包括 RGB 图像、光流和深度图。各个部分显示了用于评估 Text2Video 和 Control Generation 任务的真实数据，有助于理解 UnityVideo 的方法与效果。*

## 5.2. Evaluation Metrics
The paper uses a comprehensive set of metrics to evaluate performance across different tasks.

### 5.2.1. Video Quality Metrics
These metrics, largely from `VBench`, are perceptual and assess the visual quality and temporal coherence of generated videos.
- **Conceptual Definitions:**
    - **Subject Consistency:** Measures if the main subject retains its identity and appearance across frames.
    - **Background Consistency:** Measures if the background remains stable and does not change unnaturally.
    - **Aesthetic Quality:** Assesses the visual appeal of the video (e.g., lighting, color, composition).
    - **Imaging Quality:** Evaluates for artifacts, blurriness, or other distortions.
    - **Temporal Flickering:** Measures high-frequency, unnatural changes in brightness or color between frames.
    - **Motion Smoothness:** Assesses if object movements are fluid and not jerky.
    - **Dynamic Degree:** Quantifies the amount of motion in the video.

### 5.2.2. Depth Estimation Metrics
- **Absolute Relative Error (AbsRel):**
    - **Conceptual Definition:** This metric measures the average relative error between the predicted depth and the ground-truth depth. A lower value is better.
    - **Mathematical Formula:**
      \$
      \text{AbsRel} = \frac{1}{|N|} \sum_{p \in N} \frac{|d_p - d_p^*|}{d_p^*}
      \$
    - **Symbol Explanation:**
        - $N$: The set of all pixels in the image.
        - $d_p$: The predicted depth at pixel $p$.
        - $d_p^*$: The ground-truth depth at pixel $p$.

- **Threshold Accuracy ($\delta$):**
    - **Conceptual Definition:** This metric calculates the percentage of pixels for which the ratio between the predicted depth and the ground-truth depth is within a certain threshold. It measures the reliability of the depth predictions. A higher value is better.
    - **Mathematical Formula:** Percentage of pixels $p$ such that $\max\left(\frac{d_p}{d_p^*}, \frac{d_p^*}{d_p}\right) = \delta < \text{threshold}$. The paper uses a threshold of **1.25**.
    - **Symbol Explanation:** Same as above.

### 5.2.3. Video Segmentation Metrics
- **mean Intersection-over-Union (mIoU):**
    - **Conceptual Definition:** IoU measures the overlap between the predicted segmentation mask and the ground-truth mask for a single object class. mIoU is the average of the IoU scores across all classes. It is the standard metric for segmentation quality.
    - **Mathematical Formula:**
      \$
      \text{mIoU} = \frac{1}{C} \sum_{i=1}^{C} \text{IoU}_i = \frac{1}{C} \sum_{i=1}^{C} \frac{TP_i}{TP_i + FP_i + FN_i}
      \$
    - **Symbol Explanation:**
        - $C$: The number of classes.
        - $TP_i$: True Positives for class $i$ (pixels correctly identified as class $i$).
        - $FP_i$: False Positives for class $i$ (pixels incorrectly identified as class $i$).
        - $FN_i$: False Negatives for class $i$ (pixels of class $i$ missed by the prediction).

- **mean Average Precision (mAP):**
    - **Conceptual Definition:** In instance segmentation, mAP is a more comprehensive metric. For each class, it calculates the "Average Precision" (AP), which is the area under the precision-recall curve. mAP is the average of these AP scores over all classes. It rewards models that are good at both detecting objects (recall) and being accurate in their predictions (precision).
    - **Mathematical Formula:** The calculation is complex, but it is based on:
        - **Precision:** $\frac{TP}{TP + FP}$ (Of all positive predictions, how many were correct?)
        - **Recall:** $\frac{TP}{TP + FN}$ (Of all actual positive instances, how many were found?)
          AP is then the weighted mean of precisions at each threshold, and mAP averages this over all classes.

## 5.3. Baselines
`UnityVideo` is compared against a wide range of state-of-the-art models, each specialized for a particular task:
- **Text-to-Video Generation:** `Kling-1.6`, `OpenSora`, `Hunyuan-13B`, `Wan-2.1-13B`, and `Aether`. These represent the top open-source and commercial T2V models.
- **Controllable Generation:** `VACE` and `Full-DiT`, which are designed to generate videos from control signals.
- **Video Depth Estimation:** `DepthCrafter`, `Geo4D`, and `Aether`, which are specialized diffusion-based models for predicting depth from video.
- **Video Segmentation:** `SAMWISE` and `SeC`, which are recent models for prompt-based object segmentation in videos.

  This diverse set of baselines ensures a rigorous comparison, testing whether the unified `UnityVideo` can outperform models designed specifically for a single task.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main results are presented in Table 1, which compares `UnityVideo` against specialized state-of-the-art models across video generation, estimation, and segmentation tasks.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Tasks</th>
<th rowspan="2">Models</th>
<th colspan="4">Video Generation - VBench & UniBench Dataset</th>
<th colspan="4">Video Estimation - UniBench Dataset</th>
</tr>
<tr>
<th colspan="4">VBench</th>
<th colspan="2">Segmentation</th>
<th colspan="2">Depth</th>
</tr>
<tr>
<th></th>
<th></th>
<th>Background Consistency</th>
<th>Aesthetic Quality</th>
<th>Overall Consistency</th>
<th>Dynamic Degree</th>
<th>mIoU ↑</th>
<th>mAP↑</th>
<th>Abs Rel ↓</th>
<th>δ &lt; 1.25 ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="5">Text2Video</td>
<td>Kling1.6</td>
<td>95.33</td>
<td>60.48</td>
<td>21.76</td>
<td>47.05</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>OpenSora2</td>
<td>96.51</td>
<td>61.51</td>
<td>19.87</td>
<td>34.48</td>
<td>-</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>HunyuanVideo-13B</td>
<td>96.28</td>
<td>53.45</td>
<td>22.61</td>
<td>41.18</td>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Wan2.1-14B</td>
<td>96.78</td>
<td>63.66</td>
<td>21.53</td>
<td>34.31</td>
<td></td>
<td>-</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Aether</td>
<td>95.28</td>
<td>48.25</td>
<td>20.26</td>
<td>37.32</td>
<td></td>
<td>-</td>
<td>0.025</td>
<td>97.95</td>
</tr>
<tr>
<td rowspan="2">Controllable Generation</td>
<td>full-dit</td>
<td>95.58</td>
<td>54.82</td>
<td>20.12</td>
<td>49.50</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
</tr>
<tr>
<td>VACE</td>
<td>93.61</td>
<td>51.24</td>
<td>17.52</td>
<td>61.32</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td rowspan="2">Depth Video Reconstruction</td>
<td>depth-crafter</td>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.065</td>
<td>96.94</td>
</tr>
<tr>
<td>Geo4D</td>
<td>-</td>
<td></td>
<td></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.053</td>
<td>97.94</td>
</tr>
<tr>
<td rowspan="2">Video Segmentation</td>
<td>SAMWISE</td>
<td></td>
<td>-</td>
<td></td>
<td>-</td>
<td>62.21</td>
<td>20.12</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>SeC</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>65.52</td>
<td>22.23</td>
<td></td>
<td></td>
</tr>
<tr>
<td rowspan="2">Unified ControGen, T2V, and Estimation</td>
<td>UnityVideo (ControlGen)</td>
<td>96.04</td>
<td>54.63</td>
<td>21.86</td>
<td>64.42</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>UnityVideo (T2V/Estimation)</td>
<td>**97.44**</td>
<td>**64.12**</td>
<td>**23.57**</td>
<td>47.76</td>
<td>**68.82**</td>
<td>**23.25**</td>
<td>**0.022**</td>
<td>**98.98**</td>
</tr>
</tbody>
</table>

**Analysis:**
- **Jack-of-All-Trades, Master of All:** The most striking result is that `UnityVideo`, a single unified model, achieves state-of-the-art or highly competitive performance across *all* tested categories.
- **Text-to-Video Generation:** In T2V mode (jointly generating RGB and depth), `UnityVideo` outperforms all top baselines, including `Wan2.1-14B`, in `Background Consistency`, `Overall Consistency`, and achieves the highest `Aesthetic Quality`. This strongly suggests that learning from multiple modalities enhances the model's core generative capabilities.
- **Video Estimation:** `UnityVideo` significantly outperforms specialized models in both depth estimation and segmentation. It achieves the lowest (best) `AbsRel` of 0.022 and the highest (best) $\delta < 1.25$ of 98.98% for depth. For segmentation, it surpasses `SeC` with a `mIoU` of 68.82. This demonstrates that the synergistic knowledge gained from joint training improves the model's perceptual and reasoning abilities.
- **Controllable Generation:** The `UnityVideo (ControlGen)` version shows strong performance, particularly in `Dynamic Degree`, indicating it generates videos with more motion while maintaining high consistency.

  The qualitative results in Figure 5 further support these findings, showing `UnityVideo` generates videos with better physical realism (e.g., light refraction) and follows control signals more faithfully than competitors.

  ![该图像是一个示意图，展示了UnityVideo在视频生成中的多模态能力，包括物理理解、可控性和细粒度分割能力等。通过联合学习不同模态，UnityVideo提供了更好的视频质量和一致性，尤其在训练仅基于人类数据的情况下，能在专业模型失效的区域进行有效泛化。](images/5.jpg)
  *该图像是一个示意图，展示了UnityVideo在视频生成中的多模态能力，包括物理理解、可控性和细粒度分割能力等。通过联合学习不同模态，UnityVideo提供了更好的视频质量和一致性，尤其在训练仅基于人类数据的情况下，能在专业模型失效的区域进行有效泛化。*

## 6.2. Ablation Studies / Parameter Analysis
The ablation studies systematically validate the design choices of `UnityVideo`.

### 6.2.1. Impact of Different Modalities (Table 2)
This study compares training a baseline T2V model versus jointly training it with a single auxiliary modality (flow or depth) and with multiple modalities.

The following are the results from Table 2 of the original paper:

| | Subject Consistency | Background Consistency | Imaging Quality | Overall Consistency |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 96.51 | 96.06 | 64.99 | 23.17 |
| Only Flow | 97.82 | 97.14 | 67.34 | 23.70 |
| Only Depth | 98.13 | 97.29 | 69.09 | 23.48 |
| **Ours-Flow** | 97.97 (+1.46) | 97.19 (+1.13) | **69.36 (+4.37)** | 23.74 (+0.57) |
| **Ours-Depth** | **98.01 (+1.50)** | **97.24 (+1.18)** | 69.18 (+4.19) | **23.75 (+0.58)** |

**Analysis:** Jointly training with even one auxiliary modality (`Only Flow`, `Only Depth`) significantly improves performance over the baseline. However, training with multiple modalities (`Ours`) provides an additional boost, especially to `Imaging Quality`. This confirms that different modalities provide complementary supervisory signals that enhance each other.

### 6.2.2. Effect of Multi-Task Training (Table 3)
This study investigates the synergy between tasks by comparing models trained on a single task (`Only ControlGen` or `Only JointGen`) versus the unified multi-task approach.

The following are the results from Table 3 of the original paper:

| | Subject Consistency | Background Consistency | Temporal Flickering | Motion Smoothness |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 96.51 | 96.06 | 98.73 | 99.30 |
| Only ControlGen | 96.53 | 95.58 | 98.45 | 99.28 |
| Only JointGen | 98.01 | 97.24 | 99.10 | 99.44 |
| **Ours-ControlGen** | 96.53 (+0.02) | **96.08 (+0.02)** | **98.79 (+0.06)** | **99.38 (+0.08)** |
| **Ours-JointGen** | 97.94 (+1.43) | 97.18 (+0.63) | 99.13 (+0.40) | 99.48 (+0.18) |

**Analysis:** Training only on `ControlGen` slightly degrades performance compared to the baseline in some metrics. However, the unified approach (`Ours-ControlGen`) recovers this performance and surpasses the baseline, showing that the other tasks (estimation, joint generation) help stabilize and improve the controllable generation capability. This is strong evidence of positive cross-task knowledge transfer.

### 6.2.3. Impact of Architectural Design (Table 4)
This study validates the `In-Context Learner` and `Modality Switcher`.

The following are the results from Table 4 of the original paper:

| | Subject Consistency | Background Consistency | Temporal Flickering | Motion Smoothness |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 96.51 | 96.06 | 98.73 | 99.30 |
| w/ In-Context Learner | 97.92 | 97.08 | 99.04 | 99.42 |
| w/ Modality Switcher | 97.94 | 97.18 | 99.13 | 99.48 |
| **Ours** | **98.31** | **97.54** | **99.35** | **99.54** |

**Analysis:** Both the `In-Context Learner` and the `Modality Switcher` individually provide significant improvements over the baseline. Combining them (`Ours`) yields the best results across all metrics, confirming that they play complementary roles in enabling effective multimodal learning. The `In-Context Learner` provides semantic guidance, while the `Modality Switcher` provides architectural adaptation.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully introduces **UnityVideo**, a novel and powerful framework for unified multi-modal, multi-task video learning. By training a single diffusion transformer to simultaneously handle five auxiliary modalities (depth, flow, segmentation, skeleton, DensePose) and three distinct tasks (conditional generation, estimation, joint generation), `UnityVideo` demonstrates significant benefits. The core findings are that this unified approach leads to accelerated convergence, superior performance that often exceeds specialized models, and enhanced zero-shot generalization. The work establishes that, similar to LLMs, unifying diverse sub-modalities is a promising path toward creating more capable and "world-aware" video generation models. The contribution of the `OpenUni` dataset and `UniBench` benchmark further facilitates future research in this direction.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations and areas for future work:
- **VAE Artifacts:** The Variational Autoencoder (VAE) used to encode videos into latent space sometimes introduces visual artifacts. This could be improved by using a better VAE or fine-tuning the existing one.
- **Scaling:** The current model has 10B parameters. The authors suggest that scaling to even larger backbones and incorporating more visual modalities could lead to further emergent capabilities and a deeper world understanding.
- **Modality Diversity:** While five modalities are a significant step, the real world contains even more information (e.g., audio, thermal, material properties). Expanding the framework to include more modalities is a clear future direction.

## 7.3. Personal Insights & Critique
- **Strengths and Inspirations:**
    - The central idea of **"unification for synergy"** is compelling and well-executed. The analogy to LLMs provides a strong conceptual anchor for the work's motivation.
    - The technical solutions are elegant and practical. The **`Dynamic Task Routing`** is a clever strategy to balance multi-task learning, and the **`In-Context Learner`** is a lightweight yet powerful way to inject semantic knowledge for modality differentiation.
    - The zero-shot generalization results are particularly impressive (e.g., segmenting unseen objects, estimating skeletons for animals). This provides strong evidence that the model is learning abstract, transferable concepts rather than just surface-level correlations.
    - The contribution of a large, high-quality dataset (`OpenUni`) is a significant service to the research community.

- **Potential Issues and Critique:**
    - **Complexity and Reproducibility:** The training pipeline is complex, involving curriculum learning, dynamic task sampling, and balancing of multiple large-scale data sources. Combined with the 10B parameter model, this makes the results very difficult and expensive to reproduce for academic labs with limited computational resources.
    - **Definition of "Reasoning":** The paper claims the model has improved "reasoning" capabilities. While it clearly shows better adherence to physical principles, it's debatable whether this constitutes true reasoning or is simply a result of more robust pattern matching enabled by the diverse training data. The model likely learns strong correlations about physics but may not have a causal understanding.
    - **Generalization Limits:** The zero-shot generalization is impressive but may have its limits. For example, estimating a skeleton for a snake or an octopus would likely fail, as their structure is fundamentally different from the bipeds/quadrupeds the model has implicitly learned from. The scope of generalization needs further probing.

      Overall, `UnityVideo` is a landmark paper that sets a new direction for video generation. It moves the field beyond simply scaling up RGB data and provides a concrete, successful framework for building more holistic and intelligent world models by embracing the diversity of visual information.