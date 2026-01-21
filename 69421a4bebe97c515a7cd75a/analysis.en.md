# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is `CameraCtrl: Enabling Camera Control for Text-to-Video Generation`.

## 1.2. Authors
The authors and their affiliations are:
*   Hao He: The Chinese University of Hong Kong, Shanghai Artificial Intelligence Laboratory
*   Yinghao Xu: Stanford University
*   Yuwei Guo: The Chinese University of Hong Kong, Shanghai Artificial Intelligence Laboratory
*   Gordon Wetzstein: Stanford University
*   Bo Dai: Shanghai Artificial Intelligence Laboratory
*   Hongsheng Li: The Chinese University of Hong Kong
*   Ceyuan Yang: Shanghai Artificial Intelligence Laboratory

    Hao He is marked with an asterisk (*), indicating a primary or corresponding author. Hongsheng Li and Ceyuan Yang are marked with a dagger (†), often denoting senior authors or project leads.

## 1.3. Journal/Conference
The paper was published on arXiv, a preprint server, on `2024-04-02T16:52:41.000Z`. arXiv is a widely used platform for disseminating research quickly in fields like physics, mathematics, computer science, and more. While it is not a peer-reviewed journal or conference, papers published on arXiv are often submitted to and eventually published in reputable venues. Its influence in the relevant field (computer vision, AI) is significant as it allows researchers to share their work and receive feedback before formal publication.

## 1.4. Publication Year
2024

## 1.5. Abstract
The paper introduces `CameraCtrl`, a novel approach to enable precise camera pose control in video diffusion models, addressing the current lack of such control in existing generation models. The core methodology involves exploring effective camera trajectory parameterization, specifically using Plücker embeddings, and integrating a plug-and-play camera pose control module trained on top of an existing video diffusion model without modifying its base architecture. A comprehensive study on training datasets revealed that videos with diverse camera distributions and appearance similar to the base model's training data significantly enhance controllability and generalization. The key findings demonstrate that `CameraCtrl` achieves precise camera control with various video generation models, including general text-to-video (T2V) and image-to-video (I2V) settings, and can be combined with other visual controllers. This marks a significant advancement toward dynamic and customized video storytelling from textual and camera pose inputs.

## 1.6. Original Source Link
*   **Original Source Link:** `https://arxiv.org/abs/2404.02101`
*   **PDF Link:** `https://arxiv.org/pdf/2404.02101v2.pdf`
*   **Publication Status:** The paper is a preprint available on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation

The core problem the paper aims to solve is the **lack of precise camera pose control in existing video generation models**. While diffusion models have significantly advanced video generation, offering controllability through text or image inputs, they fall short in allowing users to specifically adjust or simulate camera viewpoints.

This problem is important because camera pose control serves as a crucial cinematic language, enabling creators to express deeper narrative nuances. In practical applications such as virtual reality, augmented reality, game development, film, and advertising industries, skillful management of camera movements is essential to emphasize emotions, highlight character relationships, and guide the audience's focus. Without this control, generated videos may lack the desired artistic and communicative impact, limiting their utility in professional content creation workflows.

Specific challenges or gaps in prior research include:
*   **Limited Camera Movement Types:** Some existing methods, like `AnimateDiff`, incorporate specific camera movements (e.g., pan, zoom) but struggle to generalize to user-customized camera trajectories.
*   **Insufficient Camera Parameter Representation:** Approaches like `MotionCtrl` condition video diffusion models on numerical camera parameters but rely solely on these values without geometric cues, leading to insufficient precision in camera control.
*   **Generalization Issues:** Many existing methods require fine-tuning parts of the base video diffusion model, which can hamper their ability to generalize camera control across different personalized video generation models or domains.
*   **Appearance Leakage:** Some control mechanisms tend to leak appearance information from the training dataset, limiting generalization by biasing the model towards the training data's aesthetics.

    The paper's entry point or innovative idea is to introduce a **precise, plug-and-play camera pose control module** that can effectively represent and inject camera pose information into existing video diffusion models without modifying their core architecture. This is achieved through:
1.  **Effective Camera Trajectory Parameterization:** Utilizing `Plücker embeddings` for camera pose representation, which provide a rich geometric interpretation for each pixel, offering a more informative description than raw numerical parameters.
2.  **Plug-and-Play Architecture:** Designing a camera control module that is `agnostic` to the appearance of the training dataset, ensuring broad applicability and generalizability across various video generation models and styles.
3.  **Data-Driven Learning:** Conducting a comprehensive study on training data to identify optimal characteristics (diverse camera distributions, similar appearance to base model data) that enhance controllability and generalization.

## 2.2. Main Contributions / Findings

The paper's primary contributions are threefold:

1.  **Introduction of `CameraCtrl` for Flexible and Precise Camera Viewpoint Control:** The paper proposes `CameraCtrl`, a novel method that empowers video diffusion models with the ability to precisely control camera viewpoints. This addresses a significant gap in existing video generation technologies, allowing for more dynamic and customized video storytelling.
2.  **Development of a Plug-and-Play Camera Control Module:** `CameraCtrl` introduces a module that can be seamlessly integrated into various existing video generation models. This module is trained on top of a video diffusion model without altering its base components, making it highly adaptable and capable of producing visually appealing camera control across different generation scenarios (e.g., general T2V, personalized T2V, I2V). The use of `Plücker embeddings` as the camera pose representation is key to its effectiveness and generalizability.
3.  **Comprehensive Analysis of Training Datasets:** The authors conducted an extensive study on the impact of various training datasets for the camera control module. They found that datasets with diverse camera distributions and appearances similar to the base model (e.g., `RealEstate10K`) yield the best trade-off between controllability and generalization. This analysis provides valuable insights for future research in this direction, guiding the selection and creation of training data for controllable video generation.

    The key conclusions or findings reached by the paper are:
*   **Superior Camera Control Accuracy:** `CameraCtrl` significantly outperforms existing methods like `MotionCtrl` and `AnimateDiff` in terms of camera control accuracy, as measured by `TransErr`, `RotErr`, and user preference studies.
*   **Preservation of Visual Quality and Dynamism:** The integration of `CameraCtrl` does not negatively impact the visual quality or dynamic degree of the generated videos, maintaining or even improving metrics like `FVD`, `CLIPSIM`, `FC`, and `ODD` compared to base models.
*   **Effectiveness of Plücker Embeddings:** `Plücker embeddings` are identified as the most effective camera representation due to their inherent geometric interpretation for each pixel, leading to superior camera control results compared to raw numerical values or other spatial representations.
*   **Optimal Architecture for Camera Encoder and Fusion:** A `T2I-Adaptor` encoder with a temporal attention module, feeding camera features into the U-Net's temporal attention layers, proved to be the most effective architecture for injecting camera conditions.
*   **Data Characteristics are Crucial:** Training on datasets with **diverse camera distributions** and **similar appearance to the base video diffusion model's training data** (e.g., `RealEstate10K`) is critical for achieving good generalizability and controllability.
*   **Broad Applicability:** `CameraCtrl` can be applied to different video generators (e.g., `AnimateDiff` for T2V, `Stable Video Diffusion` for I2V) and can be integrated with other video control methods (e.g., `SparseCtrl`), demonstrating its versatility and enhancing its application scenarios.

    These findings collectively solve the problem of lacking precise camera control, offering a robust and generalizable solution that pushes the boundaries of dynamic and customized video storytelling.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

To understand `CameraCtrl`, a reader should be familiar with the following foundational concepts:

### 3.1.1. Diffusion Models
Diffusion models are a class of generative models that have shown remarkable success in synthesizing high-quality data, particularly images and videos. The core idea behind diffusion models is a two-step process:
1.  **Forward Diffusion Process:** In this step, noise is gradually added to an input data point (e.g., an image) over a series of $T$ steps, transforming it into pure Gaussian noise. This process is typically fixed and not learned.
2.  **Reverse Diffusion Process:** This is the generative step. A neural network is trained to learn the reverse of the forward process, i.e., to progressively denoise a sample starting from pure Gaussian noise, recovering the original data point. Each step involves predicting the noise that was added in the forward process and subtracting it, guided by a trained model.

    The training objective for diffusion models often involves minimizing the mean squared error (MSE) between the predicted noise and the actual noise added. This allows the model to learn to reverse the corruption process effectively.

### 3.1.2. Text-to-Video (T2V) Generation
`Text-to-Video` generation refers to the task of creating video content directly from a textual description (a prompt). This typically involves:
*   **Text Encoding:** The input text prompt is first encoded into a numerical representation (embedding) that captures its semantic meaning. This is often done using large language models or text encoders like `CLIP`.
*   **Video Generation:** This text embedding then guides a generative model (like a diffusion model) to synthesize a sequence of frames that form a coherent video matching the text description.
*   **Temporal Consistency:** A significant challenge in T2V is ensuring that the generated frames are not only individually realistic but also flow smoothly and consistently over time, maintaining object identity and motion.

### 3.1.3. Image-to-Video (I2V) Generation
`Image-to-Video` generation is similar to T2V but takes an initial image as a primary condition, along with an optional text prompt. The goal is to generate a video sequence that starts with or is consistent with the input image. This is particularly useful for animating a static image or continuing a visual narrative.

### 3.1.4. U-Net Architecture
`U-Net` is a convolutional neural network architecture originally developed for biomedical image segmentation. It has a distinctive U-shaped structure, consisting of:
*   **Encoder (Contracting Path):** Downsamples the input through a series of convolutional and pooling layers, capturing contextual information and reducing spatial dimensions while increasing feature channels.
*   **Decoder (Expanding Path):** Upsamples the features through a series of upsampling and convolutional layers, gradually recovering spatial resolution.
*   **Skip Connections:** Crucially, `U-Net` includes skip connections that concatenate features from the encoder path to the corresponding layers in the decoder path. These connections help preserve fine-grained details lost during downsampling and improve the localization accuracy in the output.
    In diffusion models, `U-Net`s are commonly used as the `noise predictor` $\hat{\epsilon}_{\theta}$, taking a noisy latent representation and a timestep as input, and predicting the noise component.

### 3.1.5. Attention Mechanisms (Spatial and Temporal)
`Attention mechanisms` allow a model to focus on specific parts of its input when processing information.
*   **Self-Attention:** A mechanism that relates different positions of a single sequence to compute a representation of the same sequence. In `U-Net`s used for images, `spatial self-attention` allows the model to weigh the importance of different spatial locations within a single image.
*   **Cross-Attention:** A mechanism that relates two different sequences. In video diffusion models, `cross-attention` is often used to fuse information from a conditioning signal (e.g., text embedding, image features) with the visual features being processed by the `U-Net`.
*   **Temporal Attention:** Specifically for videos, `temporal attention` mechanisms are designed to capture relationships and dependencies between frames across the time dimension. This is crucial for maintaining `temporal consistency` and modeling motion in generated videos. `Temporal attention` layers allow the model to aggregate information from different frames when generating a particular frame, ensuring smooth transitions and coherent motion.

### 3.1.6. Camera Pose (Intrinsic and Extrinsic Parameters)
Camera pose describes the position and orientation of a camera in 3D space, and how it projects the 3D world onto a 2D image plane. It is typically defined by:
*   **Intrinsic Parameters ($\mathbf{K} \in \mathbb{R}^{3 \times 3}$):** These describe the internal characteristics of the camera, such as focal length ($f_x, f_y$), principal point ($c_x, c_y$), and skew coefficient. They map 3D camera coordinates to 2D pixel coordinates. A common intrinsic matrix form is:
    \$
    \mathbf{K} = \begin{pmatrix}
    f_x & s & c_x \\
    0 & f_y & c_y \\
    0 & 0 & 1
    \end{pmatrix}
    \$
    where $s$ is the skew parameter, often assumed to be 0.
*   **Extrinsic Parameters ($\mathbf{E} = [\mathbf{R}; \mathbf{t}] \in \mathbb{R}^{3 \times 4}$):** These describe the camera's position and orientation relative to a world coordinate system.
    *   **Rotation Matrix ($\mathbf{R} \in \mathbb{R}^{3 \times 3}$):** Represents the camera's orientation in 3D space (e.g., yaw, pitch, roll). It is an orthogonal matrix with determinant 1.
    *   **Translation Vector ($\mathbf{t} \in \mathbb{R}^{3 \times 1}$):** Represents the camera's position in 3D space.

### 3.1.7. Plücker Embeddings
`Plücker embeddings` (specifically, line `Plücker coordinates`) are a way to represent lines in 3D space. In the context of computer graphics and vision, they can be adapted to represent camera rays. For a pixel `(u, v)` in an image, a camera ray originates from the camera center $\mathbf{o}$ and extends in a direction $\mathbf{d}_{u,v}$ towards the 3D point projected onto that pixel. The `Plücker embedding` for such a ray is a 6D vector $\mathbf{p}_{u,v} = (\mathbf{o} \times \mathbf{d}_{u,v}, \mathbf{d}_{u,v})$.
*   $\mathbf{o} \in \mathbb{R}^3$: The camera center in world coordinates.
*   $\mathbf{d}_{u,v} \in \mathbb{R}^3$: The unit direction vector from the camera center to the pixel `(u, v)` in world coordinates.
*   $\mathbf{o} \times \mathbf{d}_{u,v}$: The cross product, which represents the moment of the line about the origin. It gives information about the plane containing the origin and the line.
    `Plücker embeddings` provide a rich, geometrically interpretable representation for each pixel's view ray, which is beneficial for tasks involving camera pose and 3D geometry.

## 3.2. Previous Works

The paper discusses several categories of previous works:

### 3.2.1. General Video Generation Models
*   **Early Diffusion Models:** $Ho et al. (2020)$, `Song et al. (2020)`, `Peebles & Xie (2023b)` established foundational work in diffusion models for image generation, which video generation methods leverage.
*   **Video Diffusion from Scratch:** `Singer et al. (2022)`, $Ho et al. (2022b)$ trained video generators from scratch, often extending 2D image diffusion architectures to accommodate video data. For example, `Video Diffusion Model (VDM)` by $Ho et al. (2022b)$ adapted a 2D architecture to video and trained it jointly on images and videos.
*   **Leveraging Pre-trained Image Generators:** Many recent works, including `AnimateDiff (Guo et al., 2023b)` and `Align-Your-Latents (Blattmann et al., 2023b)`, build upon powerful pre-trained `text-to-image (T2I)` models like `Stable Diffusion (Rombach et al., 2022)`. They inflate the 2D architecture by adding temporal layers and fine-tuning on large video datasets. `Align-Your-Latents` efficiently converts a T2I model into a video generator by aligning independently sampled noise maps. `Stable Video Diffusion (SVD)` by `Blattmann et al. (2023a)` further extends this with more elaborate training steps and data curation.
*   **Temporal Coherency Improvements:** `Lumiere (Bar-Tal et al., 2024)` improved temporal consistency by directly generating full-frame-rate videos instead of relying on temporal super-resolution.
*   **Transformer Backbones:** Works like `W.A.L.T. (Gupta et al., 2023)` and `Sora (Brooks et al., 2024)` utilize scalable `transformer` backbones and operate in compressed latent spaces. `Kondratyuk et al. (2023)` explored using discrete tokens with language models for video generation.

### 3.2.2. Controllable Video Generation
The general theme here is to move beyond text or image conditioning to more precise control signals.
*   **Structural Control Signals:**
    *   $Guo et al. (2023a)$, `Chen et al. (2023c)`, `Zhang et al. (2023c)`, `Khachatryan et al. (2023)`, $Hu et al. (2023)$, $Xu et al. (2023)$ use signals like depth maps, skeleton sequences, or canny maps to control scene or human motion.
    *   `SparseCtrl (Guo et al., 2023a)` uses sketch images as control signals to achieve high video quality and accurate temporal relationships.
*   **Camera Control Efforts:**
    *   `AnimateDiff (Guo et al., 2023b)` incorporates a `MotionLoRA` module to enable specific types of camera movement. However, it struggles with user-customized trajectories. `LoRA (Low-Rank Adaptation)` by $Hu et al. (2021)$ is a parameter-efficient fine-tuning technique that adds small, low-rank matrices to an existing model's layers, allowing adaptation to new tasks with minimal trainable parameters.
    *   `Direct-a-Video (Yang et al., 2024a)` proposes a camera embedder but is limited to controlling only three basic camera parameters (e.g., pan left).
    *   `MotionCtrl (Wang et al., 2023)` offers more flexible camera control by conditioning on a sequence of camera pose parameters. However, it relies solely on numerical values without geometric cues, limiting accuracy and generalization across different video domains because it requires fine-tuning part of the video diffusion model.

### 3.2.3. Camera Control within Related Domains
*   **Concurrent Works (from Appendix B):** The paper acknowledges concurrent research in camera control.
    *   `VD3D (Bahmani et al., 2024)`: Integrates camera control into `DiT-based` (Diffusion Transformer) models with a novel camera representation module in spatiotemporal transformers. `DiT` (e.g., `Peebles & Xie, 2023a`) replaces the `U-Net` backbone with `Transformer` blocks for scalability.
    *   `CamCo (Xu et al., 2024b)`: Leverages `epipolar constraints` for 3D consistency in image-to-video generation. `Epipolar geometry` describes the geometric relationship between two cameras viewing the same 3D scene, crucial for 3D reconstruction and multi-view consistency.
    *   `CVD (Kuang et al., 2024)`: Extends camera control for multi-view video generation with `cross-view consistency`.
    *   `Recapture (Zhang et al., 2024)`: Focuses on video-to-video camera control for modifying viewpoints in existing content, but limited to simpler scenes.
    *   `Cavia (Xu et al., 2024a)`: Enhances multi-view generation through training on diverse datasets.
    *   `Cheong et al. (2024)`: Improves camera control accuracy using a `classifier-free guidance`-like mechanism in a `DiT-based` model. `Classifier-free guidance (Ho & Salimans, 2022)` is a technique to improve the quality of samples generated by diffusion models by interpolating between conditional and unconditional noise predictions.

## 3.3. Technological Evolution

The field of generative AI for video has evolved from simple 2D image generation to increasingly complex video synthesis, with a growing demand for fine-grained control.
*   **Early Generative Models:** Initially, methods focused on static image generation (e.g., GANs, early VAEs).
*   **Introduction of Diffusion Models:** The stability and high-quality outputs of diffusion models revolutionized image generation, providing a strong foundation.
*   **Extension to Video:** Adapting diffusion models for video involved adding temporal dimensions, initially through simple extensions of 2D architectures, then by incorporating explicit temporal modeling (e.g., temporal attention layers).
*   **Addressing Temporal Coherency:** Early video diffusion models often struggled with flicker or inconsistent object identities. Subsequent works focused on improving temporal consistency through various architectural designs and training strategies.
*   **Enhancing Controllability:** As video quality improved, the focus shifted to `controllability`. Initial efforts involved text or image prompts, then evolved to more precise signals like depth, pose, and finally, camera control.
*   **Camera Control Emergence:** Recent efforts specifically target camera control, recognizing its importance for cinematic expression. `AnimateDiff` and `MotionCtrl` were early attempts, but `CameraCtrl` represents a step forward by addressing their limitations in precision and generalization.

    This paper's work fits within this timeline by pushing the boundaries of `controllability` in video generation, specifically tackling the challenging aspect of `camera pose control` with a highly effective and generalizable approach.

## 3.4. Differentiation Analysis

Compared to the main methods in related work, `CameraCtrl` offers several core differences and innovations:

*   **Compared to `AnimateDiff` (`MotionLoRA`):**
    *   **Core Difference:** `AnimateDiff` provides limited, predefined camera movements (`MotionLoRA`) and struggles to generalize to custom camera trajectories. `CameraCtrl`, however, is designed for **flexible and precise control over arbitrary user-defined camera trajectories**.
    *   **Innovation:** `CameraCtrl` achieves this by using a more geometrically rich `Plücker embedding` representation for camera poses and a dedicated `plug-and-play` module, allowing it to accurately interpret and generate complex camera movements.

*   **Compared to `MotionCtrl`:**
    *   **Core Difference:** `MotionCtrl` conditions on numerical camera parameters but lacks geometric cues, leading to less precise control and difficulty in distinguishing camera motion from scene motion. It also requires fine-tuning part of the base model, limiting its generalization across different video domains.
    *   **Innovation:**
        1.  **Geometric Representation:** `CameraCtrl` utilizes `Plücker embeddings`, which provide a pixel-wise spatial embedding with strong geometric interpretation, enabling more accurate camera control. This helps the model correlate camera values with image pixels.
        2.  **Plug-and-Play Design:** `CameraCtrl`'s module is `agnostic` to the appearance of the training dataset and leaves the base model untouched. This ensures superior `generalization ability` across various video generation models (T2V, personalized T2V, I2V) and domains, which `MotionCtrl` explicitly lacks.
        3.  **Numerical Stability:** The value ranges of `Plücker embeddings` are more uniform than raw camera parameters, which benefits the learning process.

*   **Compared to other Controllable Video Generation Methods (e.g., `SparseCtrl`, depth/skeleton controls):**
    *   **Core Difference:** While these methods control `content` (e.g., object motion, scene structure), `CameraCtrl` specifically focuses on controlling the `viewpoint` or `cinematic perspective`.
    *   **Innovation:** `CameraCtrl`'s `plug-and-play` nature allows it to **integrate with these other control methods**, enabling multi-modal control (e.g., controlling both scene structure via `SparseCtrl` and camera movement via `CameraCtrl` simultaneously), offering a more comprehensive creative toolkit.

*   **Architectural Choices and Training Strategy:**
    *   **Innovation:** `CameraCtrl`'s detailed ablation studies on camera representation, encoder architecture (using a `T2I-Adaptor` based encoder with `temporal attention`), fusion points (injecting into `temporal attention` layers), and data selection are distinct. The finding that datasets with diverse camera distributions and similar appearance to the base model are optimal is a significant contribution to understanding effective training strategies.

        In summary, `CameraCtrl` differentiates itself by offering **unprecedented precision, flexibility, and generalization** in camera pose control, achieved through a novel geometric representation, a modular plug-and-play design, and an optimized data-driven training approach, surpassing limitations of previous attempts in the field.

# 4. Methodology

The `CameraCtrl` methodology addresses three key questions for integrating precise camera control into existing video generation methods:
1.  How to effectively represent the camera condition to reflect 3D geometric movement?
2.  How to seamlessly inject this camera condition into existing video generators?
3.  What type of training data should be utilized for proper model training?

    This section will detail the approach `CameraCtrl` takes to answer these questions, starting with a brief background on video generation models, then diving into the camera pose representation, the camera control module design, and finally, the data selection process.

## 4.1. Preliminaries of Video Generation

`CameraCtrl` builds upon existing video diffusion models, which primarily extend 2D image diffusion architectures to handle video data. These models often adhere to the original formulation used for image generation, with added temporal components.

### 4.1.1. Video Diffusion Models

In video diffusion models, a sequence of $N$ images (or their latent features) is represented as $z_0^{1:N}$. Noise $\epsilon$ is gradually added to this sequence over $T$ steps, transforming it into a normal distribution. A neural network, denoted as $\hat{\epsilon}_{\theta}$, is trained to predict the added noise at each step $t$, given the noised input $z_t^{1:N}$, a condition signal $c_t$, and the timestep $t$. The training objective is to minimize the mean squared error (MSE) between the predicted noise and the ground truth noise:

\$
\mathcal{L}(\theta) = \mathbb{E}_{z_0^{1:N}, \epsilon, c_t, t} [ || \epsilon - \hat{\epsilon}_{\theta}(z_t^{1:N}, c_t, t) ||_2^2 ]
\$

Where:
*   $\mathcal{L}(\theta)$: The loss function to be minimized, parameterized by $\theta$.
*   $\theta$: The learnable parameters of the neural network $\hat{\epsilon}_{\theta}$.
*   $\mathbb{E}_{z_0^{1:N}, \epsilon, c_t, t} [ \cdot ]$: Expectation over the initial data $z_0^{1:N}$, the sampled noise $\epsilon$, the condition $c_t$, and the timestep $t$.
*   $z_0^{1:N}$: The clean (un-noised) sequence of $N$ images or their latent representations.
*   $\epsilon$: The ground truth noise added to $z_0^{1:N}$ at a specific timestep $t$.
*   $\hat{\epsilon}_{\theta}(\cdot)$: The neural network (typically a U-Net) parameterized by $\theta$, which predicts the noise.
*   $z_t^{1:N}$: The noised latent representation of the video sequence at timestep $t$.
*   $c_t$: The embedding of the corresponding condition signal (e.g., text prompts) at timestep $t$.
*   $t$: The current diffusion timestep.
*   $|| \cdot ||_2^2$: The squared $L_2$ norm, representing the mean squared error.

### 4.1.2. Controllable Video Generation

To enhance controllability beyond simple text or image conditioning, additional structural control signals $s_t$ (e.g., depth maps, canny maps) can be incorporated. These signals are typically fed into an additional encoder $\Phi_s$ and then injected into the generator. The objective function is modified to include this structural control:

\$
\mathcal{L}(\theta) = \mathbb{E}_{z_0^{1:N}, \epsilon, c_t, s_t, t} [ \| \epsilon - \hat{\epsilon}_{\theta}(z_t^{1:N}, c_t, \Phi_s(s_t), t) \|_2^2 ]
\$

Where:
*   $s_t$: The additional structural control signal at timestep $t$.
*   $\Phi_s$: An encoder that processes the structural control signal $s_t$ to produce a feature representation.
*   All other symbols are as defined above.

    In `CameraCtrl`, camera poses are treated as an additional control signal. The goal is to train a camera encoder $\Phi_c$ that processes camera pose information, strictly following the objective of Equation (2) to integrate it into the video diffusion models.

## 4.2. Camera Pose Representation

A crucial aspect of `CameraCtrl` is how camera pose is represented to accurately reflect 3D geometric movement.

### 4.2.1. Traditional Camera Representation

Traditionally, camera pose is described by:
*   **Intrinsic Parameters ($\mathbf{\bar{K}} \in \mathbb{R}^{3 \times 3}$):** These define the camera's internal optics and projection properties.
*   **Extrinsic Parameters ($\mathbf{\dot{E}} = [\mathbf{\dot{R}}; \mathbf{t}] \in \mathbb{R}^{3 \times 4}$):** These define the camera's position and orientation in world coordinates. $\mathbf{\dot{R}} \in \mathbb{R}^{3 \times 3}$ is the rotation matrix, and $\mathbf{t} \in \mathbb{R}^{3 \times 1}$ is the translation vector.

### 4.2.2. Challenges with Raw Parameter Conditioning

Directly feeding raw camera parameters (matrices, Euler angles) to the generator for conditioning has several drawbacks:
1.  **Mismatched Learning:** The rotation matrix $\mathbf{\dot{R}}$ is constrained by orthogonality, while the translation vector $\mathbf{t}$ is typically unconstrained in magnitude. This fundamental difference can lead to difficulties in the learning process for a camera control model.
2.  **Lack of Visual Correlation:** Raw numerical parameters make it hard for the model to establish a direct correlation between these values and individual image pixels, which limits precise control over visual details.
3.  **Non-pixel-wise:** Raw values or Euler angles are global parameters for the entire frame, not pixel-wise spatial embeddings, as illustrated in Figure 6.

### 4.2.3. Plücker Embeddings for Camera Pose

To overcome these challenges, `CameraCtrl` adopts `Plücker embeddings` (Sitzmann et al., 2021) as the primary form of camera pose representation. This choice is motivated by their ability to encode geometric interpretations for each pixel in a video frame, offering a comprehensive description of camera pose information.

For each pixel `(u, v)` in the image coordinate space, its `Plücker embedding` is calculated as:

\$
\mathbf{p}_{u,v} = (\mathbf{o} \times \mathbf{d}_{u,v}, \mathbf{d}_{u,v}) \in \mathbb{R}^6
\$

Where:
*   $\mathbf{p}_{u,v}$: The 6-dimensional `Plücker embedding` for the ray corresponding to pixel `(u, v)`.
*   $\mathbf{o} \in \mathbb{R}^3$: The camera center (origin) in world coordinate space.
*   $\mathbf{d}_{u,v} \in \mathbb{R}^3$: The direction vector in world coordinate space, pointing from the camera center $\mathbf{o}$ to the pixel `(u, v)`. This vector is calculated using the inverse projection process:

    \$
    \mathbf{d}_{u,v} = \mathbf{R} \mathbf{K}^{-1} [ u, v, 1 ]^T + \mathbf{t}
    \$

    Where:
    *   $\mathbf{R}$: The rotation part of the camera's extrinsic parameters.
    *   $\mathbf{K}^{-1}$: The inverse of the camera's intrinsic parameter matrix.
    *   $[u, v, 1]^T$: The homogeneous coordinates of the pixel in the image plane.
    *   $\mathbf{t}$: The translation part of the camera's extrinsic parameters.
        The resulting vector $\mathbf{d}_{u,v}$ is then normalized to ensure it has a unit length.

For the $i$-th frame in a video sequence, its `Plücker embedding` is an array $\mathbf{P}_i \in \mathbb{R}^{6 \times h \times w}$, where $h$ and $w$ are the height and width of the frame, respectively. This means each pixel in the frame has its own 6D `Plücker embedding`. The entire camera trajectory of a video is represented as a sequence of these pixel-wise embeddings: $\mathbf{P} \in \mathbb{R}^{n \times 6 \times h \times w}$, where $n$ is the total number of frames.

### 4.2.4. Advantages of Plücker Embeddings

*   **Geometric Interpretation:** `Plücker embeddings` provide a direct geometric interpretation for each pixel, offering a more informative description of camera pose information to the base video generators compared to numerical matrices. This helps in adopting the temporal consistency ability of base video generators for specific camera trajectories.
*   **Uniform Value Ranges:** The values within `Plücker embeddings` tend to have more uniform ranges, which is beneficial for the learning process of data-driven models.
*   **Pixel-wise Spatial Embedding:** Unlike raw matrices or Euler angles, `Plücker embeddings` naturally form a pixel-wise spatial map, which is crucial for spatially-aware control, as shown in Figure 6.

    The following figure (Figure 6 from the original paper) illustrates the different camera representations:

    ![Figure 6: Different camera representation. The left subfigure row shows the camera represented using the intrinsic `K _ { i }` and the extrinsic matrices `E _ { i }` (composed of rotation matrix `R _ { i }` and the translation vector `t _ { i }` ). The middle subfigure give the camera representation of converting the rotation matrix `R _ { i }` into Euler angles $\\alpha _ { i } , \\beta _ { i } , \\gamma _ { i }$ . Plücker embedding are given in the right subfigure, the intrinsic and extrinsic matrices are converted into the Plücker embeddings to form a pixel-wise spatial embedding. While the left and middle camera representations are not a pixel-wise camera representations naturally.](images/6.jpg)
    *该图像是示意图，展示了不同的相机表示方法。左侧子图展示了利用内部矩阵 $K_i$ 和外部矩阵 $E_i$（由旋转矩阵 $R_i$ 和位移向量 $t_i$ 组成）的相机表示。中间子图则展示了将旋转矩阵 $R_i$ 转换为欧拉角 $\alpha_i, \beta_i, \gamma_i$ 的相机表示。右侧子图给出了普吕克嵌入，将内部和外部矩阵转换为像素级的空间嵌入。*

## 4.3. Camera Controllability into Video Generators

After establishing the `Plücker embedding` as the camera representation, the next step is to design a mechanism to integrate this information into existing video diffusion models. `CameraCtrl` follows the approach of using an encoder to extract features from the `Plücker embedding` sequence and then fusing these features into the video generator's `U-Net` architecture.

The following figure (Figure 2 from the original paper) illustrates the framework of CameraCtrl:

![Figure 2: Framework of CameraCtrl. (a) Given a pre-trained video diffusion model (e.g. AnimateDiff (Guo et al., 2023b)) and SVD (Blattmann et al., 2023a), CameraCtr1 trains a camera encoder on it, which takes the Plücker embeding as input and outputs multi-scale camera representations. These features are then integrated into the temporal attention layers of the U-Net at their respective scales to control the video generation process. (b) Details of the camera injection process. The camera features `c _ { t }` and the latent features `z _ { t }` are first combined through the element-wise addition. A learnable linear layer is adopted to further fuse two representations which are then fed into the first temporal attention layer of each temporal block.](images/2.jpg)
*该图像是示意图，展示了CameraCtrl的框架。图(a)中显示了一个预训练的视频扩散模型及其上训练的相机编码器，该编码器处理Plücker嵌入并生成多尺度相机表示，这些表示被整合到U-Net的时间注意力层中以控制视频生成过程。图(b)详细描述了相机特征$c_t$和潜在特征$z_t$的注入过程，通过元素级相加后，融合到一个线性层中，最终输入到每个时间块的首个时间注意力层。*

### 4.3.1. Camera Encoder ($\Phi_c$)

The camera encoder's role is to process the `Plücker embedding` sequence and extract multi-scale features that can guide the video generation process.

*   **Input Choice:** `CameraCtrl` specifically designs its camera encoder $\Phi_c$ to **only take the `Plücker embedding` as input**, and **not** combine it with image features or noised latents (like `ControlNet` does). This design choice is critical because empirical analysis showed that using the input image's latent representation (as in `ControlNet`) tends to "leak" appearance information from the training dataset. This leakage biases the model towards the training data's aesthetics, limiting its generalization ability to control camera poses across various domains. By isolating the camera information, $\Phi_c$ becomes agnostic to the appearance of the training dataset.
*   **Architecture:** The camera encoder is based on the `T2I-Adaptor` encoder (Mou et al., 2023) but is adapted for videos. It includes a `temporal attention module` after each convolutional block. This addition allows the encoder to capture temporal relationships between camera poses throughout the video clip, which is essential for consistent camera movement.
    *   **Detailed Architecture (from Appendix D.1):**
        The camera encoder consists of:
        1.  A `pixel unshuffle layer`: This layer typically decreases spatial resolution while increasing channel depth, effectively processing the input $\mathbf{P} \in \mathbb{R}^{b \times n \times 6 \times h \times w}$ (batch size $b$, number of frames $n$, 6 channels for `Plücker`, height $h$, width $w$).
        2.  A `convolution layer`: A 3x3 convolution layer further processes the features.
        3.  **Four encoder scales:** Each scale processes features at decreasing spatial resolutions and increasing channel depths.
            *   Each encoder scale (except the first) is composed of one `downsample ResNet block` and one `ResNet block`.
            *   Crucially, **each `ResNet block` is followed by one `temporal attention block`**. This block is vital for capturing temporal dependencies.
    *   **Temporal Attention Block Structure:** A `temporal attention block` consists of a `temporal self-attention layer`, `layer normalizations`, and a `position-wise MLP`. The process is as follows:

        \$
        \begin{array}{rl}
        & \zeta \gets x + \mathrm{PosEmb}(x) \\
        & \zeta_1 \gets \mathrm{LayerNorm}(\zeta) \\
        & \zeta_2 \gets \mathrm{MultiHeadSelfAttention}(\zeta_1) + \zeta \\
        & \zeta_3 \gets \mathrm{LayerNorm}(\zeta_2) \\
        & x \gets \mathrm{MLP}(\zeta_3) + \zeta_2
        \end{array}
        \$
        Where:
        *   $x$: Input feature map.
        *   $\mathrm{PosEmb}(x)$: Temporal positional embedding added to the input features to inject temporal order information.
        *   $\zeta$: Features with positional embeddings.
        *   $\mathrm{LayerNorm}(\cdot)$: Layer Normalization, which normalizes the activations across features.
        *   $\mathrm{MultiHeadSelfAttention}(\cdot)$: The multi-head self-attention mechanism, which allows the model to weigh different frames' contributions when processing a specific frame in the sequence.
        *   $\mathrm{MLP}(\cdot)$: A Multi-Layer Perceptron (feed-forward network).
    *   **Output:** The camera encoder delivers multi-scale features, where the channel numbers of these features ($c_1, c_2, c_3, c_4$) are designed to match the corresponding `U-Net` output feature channels at the same resolutions.

### 4.3.2. Camera Fusion

After extracting multi-scale camera features, these features are seamlessly integrated into the `U-Net` architecture of the video diffusion model.

*   **Injection Point:** `CameraCtrl` injects the camera features into the **temporal attention blocks** of the `U-Net`. This decision is based on the understanding that camera motion primarily induces `global view changes across frames`, which aligns well with the `temporal attention layer`'s capability to capture temporal relationships and the inherent sequential nature of a camera trajectory. In contrast, `spatial attention layers` focus on individual frames.
*   **Fusion Process (illustrated in Figure 2(b)):**
    1.  The image latent features ($z_t$) and the camera pose features ($c_t$) (output from $\Phi_c$) are first combined through **element-wise addition**.
    2.  The integrated feature is then passed through a `learnable linear layer`.
    3.  The output of this linear layer is directly fed into the **fixed first temporal attention layer** of each `temporal attention module` within the `U-Net`. This ensures that camera information guides the temporal reasoning of the video generation process.

        The ablation study confirmed that injecting camera features into the temporal attention layers yields the best results (Table 2c in the paper). Furthermore, injecting features into both the U-Net encoder and decoder (rather than just the encoder) improved camera control accuracy (Table 4 in Appendix F.1).

## 4.4. Learning Camera Distribution in a Data-Driven Manner

Training the camera encoder and fusion layers requires a substantial dataset of videos with accurate camera pose annotations. Camera trajectories can be obtained either by `structure-from-motion (SfM)` techniques (e.g., `COLMAP`) for realistic videos or from rendering engines (e.g., Blender) for synthetic data. `CameraCtrl` investigates the impact of various training data types on the camera-controlled generator.

### 4.4.1. Dataset Selection

The primary goal for dataset selection is to choose data that:
1.  Has **appearances that closely match** the training data of the base video diffusion models (e.g., `WebVid10M`).
2.  Exhibits the **widest possible camera pose distribution**.

    Three candidate datasets were considered:
*   **Objaverse (Deitke et al., 2023):** Contains computer-generated imagery (CGI) with diverse camera distributions (as camera parameters can be controlled during rendering). However, its appearance (objects against white backgrounds) significantly differs from real-world datasets used for base models (e.g., `WebVid10M`). This `distribution gap` can cause the model to `leak appearance information` and limit generalization.
*   **MVImageNet (Yu et al., 2023):** A real-world dataset with some backgrounds and complex individual camera trajectories. However, most of its camera trajectories are limited to horizontal rotations, lacking broader diversity.
*   **RealEstate10K (Zhou et al., 2018):** A real-world dataset featuring indoor and outdoor scenes. It offers `complex individual camera trajectories` and a `considerable variety among different camera trajectories` (diverse camera pose distribution). Its appearance is also more aligned with `WebVid10M`. This dataset was ultimately selected as the primary training dataset due to its optimal balance of appearance similarity and camera diversity.
    Other similar datasets like `ACID (Liu et al., 2021)` and `MannequinChallenge (Li et al., 2019)` were also considered but found to have smaller data volumes and did not improve performance when combined with `RealEstate10K`.

The following figure (Figure 5 from the original paper) shows samples from the different datasets:

![Figure 5: Samples of different datasets. Rows 1 to row 3 are samples from the Objaverse dataset, which has random camera poses for each rendered image. Rows 4 to row 6 show the samples from the MVImageNet dataset. Samples of the RealEstate10K dataset are presented from rows 7 to row 9.](images/5.jpg)
*该图像是示意图，展示了不同的数据集样本。第一至第三行为 Objaverse 数据集样本，随机相机视角；第四至第六行为 MVImageNet 数据集；第七至第九行为 RealEstate10K 数据集样本。*

### 4.4.2. Measuring Camera Controllability

To monitor training and evaluate camera control quality, `CameraCtrl` introduces two metrics quantifying the error between input camera conditions and the camera trajectory of generated videos. `COLMAP (Schönberger & Frahm, 2016)` is used to extract camera pose sequences from generated videos.

*   **RotErr (Rotation Error):** Measures the angular difference between the ground truth rotation matrices and the generated rotation matrices.

    \$
    \mathrm{RotErr} = \sum_{i=1}^{n} \arccos\left(\frac{\mathrm{tr}(\mathbf{R}_{\mathrm{gen}}^i (\mathbf{R}_{\mathrm{gt}}^i)^T) - 1}{2}\right)
    \$
    Where:
    *   $\mathrm{RotErr}$: The total rotation error over all frames.
    *   $n$: Total number of frames in the video clip.
    *   $\arccos(\cdot)$: The arccosine function, used to convert a dot product or trace value into an angle.
    *   $\mathrm{tr}(\cdot)$: The trace of a matrix (sum of diagonal elements).
    *   $\mathbf{R}_{\mathrm{gen}}^i$: The generated rotation matrix for the $i$-th frame.
    *   $\mathbf{R}_{\mathrm{gt}}^i$: The ground truth rotation matrix for the $i$-th frame.
    *   $(\mathbf{R}_{\mathrm{gt}}^i)^T$: The transpose of the ground truth rotation matrix.
        This formula is derived from the property that for two rotation matrices $\mathbf{A}$ and $\mathbf{B}$, the angle $\theta$ between them can be found using $\mathrm{tr}(\mathbf{A}\mathbf{B}^T) = 1 + 2\cos\theta$.

*   **TransErr (Translation Error):** Measures the $L_2$ distance (Euclidean distance) between the ground truth translation vectors and the generated translation vectors.

    \$
    \mathrm{TransErr} = \sum_{j=1}^{n} \| \mathbf{T}_{\mathrm{gt}}^i - \mathbf{T}_{\mathrm{gen}}^i \|_2^2
    \$
    Where:
    *   $\mathrm{TransErr}$: The total translation error over all frames.
    *   $n$: Total number of frames.
    *   $\| \cdot \|_2^2$: The squared Euclidean norm (L2 norm).
    *   $\mathbf{T}_{\mathrm{gt}}^i$: The ground truth translation vector for the $i$-th frame.
    *   $\mathbf{T}_{\mathrm{gen}}^i$: The generated translation vector for the $i$-th frame.

        Special considerations for `COLMAP` and these metrics (from Appendix D.5):
*   `COLMAP` can be unstable for short video clips (16 frames in T2V, 14 frames in I2V), so failed extractions are manually filtered.
*   `COLMAP` is scale-invariant, meaning it determines relative scale but not absolute scale. This affects `TransErr`. To mitigate this, a post-processing step normalizes the scale: relative poses are computed (setting the first frame's extrinsic matrix as identity), and a rescale factor is derived from the translation gap between the first two frames of both generated and ground truth trajectories. This factor is then applied to align the scales.

# 5. Experimental Setup

## 5.1. Datasets

The experiments primarily use `RealEstate10K (Zhou et al., 2018)` as the training dataset for `CameraCtrl`, with comparisons against `Objaverse (Deitke et al., 2023)` and `MVImageNet (Yu et al., 2023)` during ablation studies. The base video diffusion models are `AnimateDiff V3 (Guo et al., 2023b)` for Text-to-Video (T2V) and `Stable Video Diffusion (SVD) (Blattmann et al., 2023a)` for Image-to-Video (I2V). The `WebVid10M (Bain et al., 2021)` dataset is often used for training base video models and as a reference for evaluation metrics.

### 5.1.1. Training Dataset: RealEstate10K
*   **Source & Characteristics:** `RealEstate10K` consists of real-world videos, primarily showcasing indoor and outdoor scenes from real estate listings. It features diverse and complex camera trajectories (e.g., pans, tilts, dollies, tracks).
*   **Scale:** Approximately `65K` video clips are used for training `CameraCtrl`.
*   **Domain:** Real-world environments, often with realistic textures and lighting.
*   **Choice Justification:** Chosen because its appearance closely resembles the training data of base video diffusion models like `WebVid10M`, and it offers a wide variety of complex camera trajectories, which is crucial for learning generalizable camera control.

### 5.1.2. Comparison Datasets: Objaverse & MVImageNet
*   **Objaverse:**
    *   **Source & Characteristics:** `Objaverse` contains a vast collection of annotated 3D objects, often rendered in synthetic environments (e.g., objects against white backgrounds). It can generate highly complex camera poses programmatically.
    *   **Domain:** Synthetic, 3D object-centric.
    *   **Choice Justification:** Used to investigate the impact of appearance distribution gap. While it has diverse camera poses, its synthetic appearance differs significantly from real-world video datasets, limiting its generalization for models trained on real data.
*   **MVImageNet:**
    *   **Source & Characteristics:** `MVImageNet` is a large-scale dataset of multi-view images. While it has some backgrounds and complex individual camera trajectories, its camera movements are often limited to horizontal rotations, lacking broader diversity in trajectory types.
    *   **Domain:** Real-world, multi-view images.
    *   **Choice Justification:** Used to study the impact of camera trajectory diversity. It has a more realistic appearance than `Objaverse` but less diverse camera trajectories than `RealEstate10K`.

### 5.1.3. Base Model Training Data Reference: WebVid10M
*   **Source & Characteristics:** `WebVid10M` is a large-scale dataset of short video clips paired with text descriptions, often collected from the web. It's commonly used to train large video diffusion models.
*   **Domain:** Diverse real-world videos.
*   **Relevance:** `CameraCtrl` aims for its training data to have an appearance similar to `WebVid10M` to ensure better generalization when integrated with models trained on it.

### 5.1.4. Data Samples
The following figure (Figure 5 from the original paper) shows examples from the three main datasets discussed for training:

![Figure 5: Samples of different datasets. Rows 1 to row 3 are samples from the Objaverse dataset, which has random camera poses for each rendered image. Rows 4 to row 6 show the samples from the MVImageNet dataset. Samples of the RealEstate10K dataset are presented from rows 7 to row 9.](images/5.jpg)
*该图像是示意图，展示了不同的数据集样本。第一至第三行为 Objaverse 数据集样本，随机相机视角；第四至第六行为 MVImageNet 数据集；第七至第九行为 RealEstate10K 数据集样本。*

*   **Objaverse (Rows 1-3):** Shows 3D rendered objects (e.g., a car, a teapot, an alien head) against simple, often white, backgrounds. The camera poses for each rendered image can be random.
*   **MVImageNet (Rows 4-6):** Displays images of real-world objects (e.g., a statue, a chair, a plant) from multiple viewpoints, often against studio or simple backgrounds. Camera trajectories are typically horizontal rotations.
*   **RealEstate10K (Rows 7-9):** Features frames from real estate videos, showing indoor (e.g., kitchen, living room) and outdoor (e.g., building exterior, garden) scenes. Camera movements are dynamic and varied.

## 5.2. Evaluation Metrics

To comprehensively evaluate `CameraCtrl`, the authors use a suite of metrics covering visual quality, text-video alignment, temporal consistency, object dynamism, and camera control accuracy.

### 5.2.1. Visual Quality & Text-Video Alignment & Temporal Consistency

*   **Fréchet Video Distance (FVD)**
    *   **Conceptual Definition:** FVD is a metric used to evaluate the realism and quality of generated videos, similar to Fréchet Inception Distance (FID) for images. It measures the "distance" between the feature distributions of real and generated videos. A lower FVD score indicates that the generated videos are more realistic and closer to the distribution of real videos.
    *   **Mathematical Formula:** The FVD is calculated based on the Fréchet distance between two multivariate Gaussian distributions:
        \$
        \mathrm{FVD}(\mathcal{R}, \mathcal{G}) = ||\mu_{\mathcal{R}} - \mu_{\mathcal{G}}||_2^2 + \mathrm{Tr}(\Sigma_{\mathcal{R}} + \Sigma_{\mathcal{G}} - 2(\Sigma_{\mathcal{R}}\Sigma_{\mathcal{G}})^{1/2})
        \$
    *   **Symbol Explanation:**
        *   $\mathcal{R}$: The set of real videos.
        *   $\mathcal{G}$: The set of generated videos.
        *   $\mu_{\mathcal{R}}$: The mean of the feature embeddings for real videos.
        *   $\mu_{\mathcal{G}}$: The mean of the feature embeddings for generated videos.
        *   $\Sigma_{\mathcal{R}}$: The covariance matrix of the feature embeddings for real videos.
        *   $\Sigma_{\mathcal{G}}$: The covariance matrix of the feature embeddings for generated videos.
        *   $||\cdot||_2^2$: The squared Euclidean norm (L2 norm).
        *   $\mathrm{Tr}(\cdot)$: The trace of a matrix.
        *   $(\cdot)^{1/2}$: The matrix square root.
    *   **Note:** Video features are typically extracted using a pre-trained 3D convolutional network (e.g., a video inception network).

*   **CLIPSIM (CLIP Score for Similarity)**
    *   **Conceptual Definition:** CLIPSIM (or CLIP Score) measures the similarity between the generated video and the input text prompt. It leverages the `CLIP (Contrastive Language-Image Pre-training)` model, which maps images and text into a shared embedding space. A higher CLIPSIM score indicates better alignment between the video content and the text description.
    *   **Mathematical Formula:** The CLIP score for a video-text pair is typically defined as the cosine similarity between their respective CLIP embeddings:
        \$
        \mathrm{CLIPSIM}(\mathbf{V}, \mathbf{T}) = \cos(\mathbf{E}_{\mathrm{CLIP}}(\mathbf{V}), \mathbf{E}_{\mathrm{CLIP}}(\mathbf{T})) = \frac{\mathbf{E}_{\mathrm{CLIP}}(\mathbf{V}) \cdot \mathbf{E}_{\mathrm{CLIP}}(\mathbf{T})}{||\mathbf{E}_{\mathrm{CLIP}}(\mathbf{V})|| \cdot ||\mathbf{E}_{\mathrm{CLIP}}(\mathbf{T})||}
        \$
    *   **Symbol Explanation:**
        *   $\mathbf{V}$: The generated video.
        *   $\mathbf{T}$: The input text prompt.
        *   $\mathbf{E}_{\mathrm{CLIP}}(\cdot)$: The `CLIP` encoder that maps videos (or individual frames averaged) and text into a shared embedding space.
        *   $\cos(\cdot, \cdot)$: The cosine similarity function.
        *   $\cdot$: Dot product.
        *   $||\cdot||$: L2 norm (magnitude of the vector).
    *   **Note:** For videos, `CLIP` embeddings are often derived by averaging the embeddings of individual frames or processing them with a temporal aggregator.

*   **Frame Consistency (FC)**
    *   **Conceptual Definition:** Frame Consistency measures the temporal coherence within a generated video, i.e., how consistent the visual content (e.g., objects, background) remains across consecutive frames. Higher FC indicates better temporal stability and less flickering.
    *   **Mathematical Formula:** While the paper does not explicitly state the formula, `Frame Consistency` metrics often rely on measuring similarity between consecutive frames' features. A common approach involves `CLIP` features:
        \$
        \mathrm{FC} = \frac{1}{N-1} \sum_{i=1}^{N-1} \cos(\mathbf{E}_{\mathrm{CLIP}}(\text{Frame}_i), \mathbf{E}_{\mathrm{CLIP}}(\text{Frame}_{i+1}))
        \$
    *   **Symbol Explanation:**
        *   $N$: The total number of frames in the video.
        *   $\text{Frame}_i$: The $i$-th frame of the generated video.
        *   $\mathbf{E}_{\mathrm{CLIP}}(\cdot)$: The `CLIP` encoder, used here to extract features for individual frames.
        *   $\cos(\cdot, \cdot)$: The cosine similarity function.

### 5.2.2. Object Dynamism

*   **Object Dynamic Degree (ODD)**
    *   **Conceptual Definition:** ODD evaluates the extent of object motion within a generated video. It aims to quantify how dynamic the objects are, distinguishing static scenes from those with active object movement. A higher ODD implies more significant object motion.
    *   **Methodology (from Appendix D.4):**
        1.  `Grounded-SAM-2 (Ren et al., 2024)` is used to segment the main object in a video. `Grounded-SAM-2` is a multi-modal foundational model capable of grounding various inputs (text, point, box) to segment anything.
        2.  `RAFT (Recurrent All-Pairs Field Transforms) (Teed & Deng, 2020)` is used to estimate `optical flow` for the entire video. `Optical flow` is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between the observer and the scene.
        3.  Only the estimated optical flows belonging to the `main object` (segmented by `Grounded-SAM-2`) are kept.
        4.  Following the "dynamic degree" metric in `VBench (Huang et al., 2023)`, these object-specific optical flows are used to determine if the video is "static" or "non-static" in terms of object motion.
        5.  The final ODD score is calculated as the `proportion of non-static videos` generated by the model.

### 5.2.3. Camera Control Accuracy

*   **RotErr (Rotation Error)**
    *   **Conceptual Definition:** Measures the average angular difference in rotation between the ground truth camera pose and the generated camera pose for each frame. Lower values indicate more accurate rotation control.
    *   **Mathematical Formula:**
        \$
        \mathrm{RotErr} = \sum_{i=1}^{n} \arccos\left(\frac{\mathrm{tr}(\mathbf{R}_{\mathrm{gen}}^i (\mathbf{R}_{\mathrm{gt}}^i)^T) - 1}{2}\right)
        \$
    *   **Symbol Explanation:**
        *   $n$: Total number of frames.
        *   $\mathbf{R}_{\mathrm{gen}}^i$: Generated rotation matrix for frame $i$.
        *   $\mathbf{R}_{\mathrm{gt}}^i$: Ground truth rotation matrix for frame $i$.
        *   $\mathrm{tr}(\cdot)$: Trace of a matrix.
        *   $\arccos(\cdot)$: Arccosine function.

*   **TransErr (Translation Error)**
    *   **Conceptual Definition:** Measures the average squared Euclidean distance between the ground truth camera translation and the generated camera translation for each frame. Lower values indicate more accurate translation control.
    *   **Mathematical Formula:**
        \$
        \mathrm{TransErr} = \sum_{j=1}^{n} \| \mathbf{T}_{\mathrm{gt}}^i - \mathbf{T}_{\mathrm{gen}}^i \|_2^2
        \$
    *   **Symbol Explanation:**
        *   $n$: Total number of frames.
        *   $\mathbf{T}_{\mathrm{gt}}^i$: Ground truth translation vector for frame $i$.
        *   $\mathbf{T}_{\mathrm{gen}}^i$: Generated translation vector for frame $i$.
        *   $\| \cdot \|_2^2$: Squared $L_2$ norm.

    *   **Note on Scale Invariance:** Since the camera pose extraction tool `COLMAP` is scale-invariant, `TransErr` requires a post-processing step (explained in Methodology Section 4.4.2) to normalize the scale of generated camera poses against ground truth to ensure meaningful comparison.

### 5.2.4. User Preference Rate
*   **Conceptual Definition:** Measures the subjective preference of human users regarding the camera control quality. Users are asked to compare videos generated by different methods against a target camera trajectory and indicate which video better aligns with the desired movement. A higher rate indicates better subjective performance.
*   **Methodology (from Appendix E):**
    *   For `AnimateDiff` (which has predefined movements), users compared against base camera movements.
    *   For `MotionCtrl` and `CameraCtrl`, users compared against complex camera trajectories extracted from `RealEstate10K` test set.
    *   50 users participated, and the tasks were designed to be as simple as possible to ensure reliable results.

## 5.3. Baselines

`CameraCtrl` is compared against two main baseline methods for camera control in video generation:

*   **AnimateDiff (Guo et al., 2023b) with MotionLoRA:**
    *   **Description:** `AnimateDiff` is a `plug-in Motion Module` that enables high-quality animation creation on personalized image backbones. It can incorporate `MotionLoRA` (Low-Rank Adaptation) to achieve specific, basic camera movements (e.g., pan, zoom).
    *   **Representativeness:** It's a widely recognized and effective text-to-video generation model, and its `MotionLoRA` provides a direct, albeit limited, camera control mechanism.
    *   **Limitation:** It supports only a few predefined camera movements and struggles with user-customized camera trajectories, making it difficult to calculate `RotErr` and `TransErr` for arbitrary paths. Comparisons are thus primarily qualitative or through user studies.

*   **MotionCtrl (Wang et al., 2023):**
    *   **Description:** `MotionCtrl` is a method that offers more flexible camera control by conditioning video diffusion models on sequences of camera pose parameters. It aims to control camera viewpoints by taking more camera parameters as input.
    *   **Representativeness:** It's a direct competitor focusing on flexible camera control using camera parameters.
    *   **Limitation:** Relies solely on numerical values of camera parameters without geometric cues, potentially leading to less precise control. It also requires fine-tuning parts of the base video diffusion model, which can limit its generalization across different video domains.
    *   **Variants:** `MotionCtrl` is evaluated with `VideoCrafter (Chen et al., 2023a)` as its base model (denoted `MotionCtrlvc`) and `Stable Video Diffusion (SVD)` as its base model (denoted `MotionCtrlSVD`).

        Additionally, the performance of `CameraCtrl` is implicitly compared against the **base video diffusion models themselves** (`AnimateDiff` and `SVD`) to ensure that adding camera control does not degrade the core video generation quality (FVD, CLIPSIM, FC, ODD).

## 5.4. Implementation Details

### 5.4.1. Base Video Diffusion Models
*   **Text-to-Video (T2V) Setting:** `AnimateDiff V3 (Guo et al., 2023b)` is used as the base model. Its ability to integrate with various T2I LoRAs (Low-Rank Adaptation modules) or base models across different genres helps evaluate `CameraCtrl`'s generalization.
*   **Image-to-Video (I2V) Setting:** `Stable Video Diffusion (SVD) (Blattmann et2al., 2023a)` is used as the base model.

### 5.4.2. Training
*   **Optimizer:** `AdamW` optimizer is used.
*   **Learning Rate:**
    *   T2V: $1 \times 10^{-4}$
    *   I2V: $3 \times 10^{-5}$
*   **Batch Size:** 32 (T2V) or 1 (I2V) per GPU.
*   **Steps:** 50K steps for both settings.
*   **Weight Decay:** 0.01
*   **Betas:** $\beta_1 = 0.9$, $\beta_2 = 0.99$
*   **Dataset:** `RealEstate10K` (around `65K` video clips).
*   **Text Prompts:** Generated using `LAVIS (Li et al., 2023)` for each video clip in the dataset.
*   **T2V Specifics:**
    *   For `AnimateDiff` base, an image LoRA is first trained on `RealEstate10K` images to help the camera control model focus on poses. This LoRA can be removed after `CameraCtrl` training.
    *   Sampled 16 images from one video clip with a stride of 8.
    *   Resolution: $256 \times 384$.
    *   Data Augmentation: Random horizontal flip for images and poses with 50% probability.
    *   Noise Schedule: Linear beta noise schedule ($\beta_{start} = 0.00085$, $\beta_{end} = 0.012$, $T = 1000$).
    *   Hardware: 16 NVIDIA A100 GPUs (80G VRAM), taking about 25 hours.
*   **I2V Specifics:**
    *   Directly trained the camera encoder and merge linear layer on top of `SVD`.
    *   Sampled 14 images from one video clip with a stride of 8.
    *   Resolution: $320 \times 576$.
    *   Noise Schedule: `EDM (Karras et al., 2022)` noise scheduler, with hyper-parameters set equal to `SVD`.
    *   Hardware: 32 NVIDIA A100 GPUs (80G VRAM), taking about 40 hours.

### 5.4.3. Inference
*   **Camera Trajectory Extraction/Design:** `COLMAP (Schönberger & Frahm, 2016)` can be used to extract camera trajectories from existing videos, or custom trajectories can be designed.
*   **Guidance Scales:** Different guidance scales are used for different video domains.
*   **Denoise Steps:** Constant 25 denoise steps for all videos.

# 6. Results & Analysis

This section analyzes the experimental results, covering quantitative comparisons, qualitative comparisons, and comprehensive ablation studies to validate `CameraCtrl`'s effectiveness.

## 6.1. Core Results Analysis

The quantitative comparison (Table 1) demonstrates `CameraCtrl`'s superiority in camera control accuracy while maintaining or improving visual quality and dynamism.

The following are the results from Table 1 of the original paper:

<table><tr><td>Method</td><td>FVD ↓</td><td>CLIPSIM ↑</td><td>FC↑</td><td>ODD ↑</td><td>TransErr↓</td><td>RotErr↓</td><td>User Preference Rate ↑ (%)</td></tr><tr><td>AnimateDiff</td><td>1022.4</td><td>0.298</td><td>0.930</td><td>56.4</td><td>Incapable</td><td>Incapable</td><td>19.4</td></tr><tr><td>MotionCtrlvc</td><td>1123.2</td><td>0.286</td><td>0.922</td><td>42.3</td><td>1402</td><td>1.58</td><td>37.0</td></tr><tr><td>CameraCtrlAD</td><td>1088.9</td><td>0.301</td><td>0.941</td><td>49.8</td><td>12.98</td><td>1.29</td><td>43.6</td></tr><tr><td>SVD</td><td>371.2</td><td>0.312</td><td>0.957</td><td>47.5</td><td>Incapable</td><td>Incapable</td><td>Incapable</td></tr><tr><td>MotionCtrlSVD</td><td>386.2</td><td>0.303</td><td>0.953</td><td>41.8</td><td>10.21</td><td>1.41</td><td>26.9</td></tr><tr><td>CameraCtrlsvD</td><td>360.3</td><td>0.298</td><td>0.960</td><td>46.5</td><td>9.02</td><td>1.18</td><td>73.1</td></tr></table>

**Comparison with Baselines:**

*   **Camera Control Accuracy (TransErr, RotErr, User Preference Rate):**
    *   **`CameraCtrlAD` (CameraCtrl with `AnimateDiff`)**: Achieves `TransErr` of 12.98 and `RotErr` of 1.29. Its `User Preference Rate` is 43.6%.
    *   **`MotionCtrlvc` (MotionCtrl with `VideoCrafter`)**: Shows significantly higher errors (`TransErr` 1402, `RotErr` 1.58) and lower `User Preference Rate` (37.0%). The vastly larger `TransErr` for `MotionCtrlvc` suggests a fundamental issue in accurately reproducing translation.
    *   **`AnimateDiff`**: Is "Incapable" of these metrics as it doesn't support custom trajectories, but its `User Preference Rate` for predefined movements is the lowest at 19.4%. This highlights `CameraCtrl`'s ability to handle user-defined and complex trajectories.
    *   **`CameraCtrlSVD` (CameraCtrl with `SVD`)**: Exhibits the lowest errors (`TransErr` 9.02, `RotErr` 1.18) and the highest `User Preference Rate` (73.1%) among all methods in the I2V setting.
    *   **`MotionCtrlSVD` (MotionCtrl with `SVD`)**: Also shows higher errors (`TransErr` 10.21, `RotErr` 1.41) and a much lower `User Preference Rate` (26.9%) compared to `CameraCtrlSVD`.
    *   **Conclusion:** `CameraCtrl` consistently outperforms `MotionCtrl` in both T2V and I2V settings across all camera control metrics, demonstrating its superior precision and alignment with desired camera movements. The large difference in `User Preference Rate` (43.6% vs 37.0% for T2V, and 73.1% vs 26.9% for I2V) is particularly compelling, indicating a much better subjective experience for users.

*   **Visual Quality & Dynamism (FVD, CLIPSIM, FC, ODD):**
    *   **Comparison to Base Models (`AnimateDiff`, `SVD`):**
        *   `CameraCtrlAD` (FVD 1088.9, CLIPSIM 0.301, FC 0.941, ODD 49.8) shows comparable or even slightly better performance on these metrics than the base `AnimateDiff` (FVD 1022.4, CLIPSIM 0.298, FC 0.930, ODD 56.4). The slight increase in FVD might be due to the added complexity of control, but other metrics show improvement.
        *   `CameraCtrlSVD` (FVD 360.3, CLIPSIM 0.298, FC 0.960, ODD 46.5) maintains very similar FVD, CLIPSIM, and ODD to the base `SVD` (FVD 371.2, CLIPSIM 0.312, FC 0.957, ODD 47.5), and even improves `FC`. This indicates that `CameraCtrl` integrates well without sacrificing the core generation capabilities of the base models.
    *   **Comparison to `MotionCtrl`:**
        *   `CameraCtrlAD` generally outperforms `MotionCtrlvc` in visual quality and dynamism metrics.
        *   `CameraCtrlSVD` also generally performs better or comparably than `MotionCtrlSVD`.
    *   **Conclusion:** `CameraCtrl` successfully adds precise camera control without negatively impacting the visual quality, text-video alignment, temporal consistency, or object dynamism of the generated videos. This is a crucial advantage for practical applications.

**Qualitative Comparison (Figure 3):**
The following figure (Figure 3 from the original paper) shows qualitative comparisons:

![Figure 3: Qualitative comparisons between CameraCtrl and MotionCtrl. The first two rows are in the T2V setting, representing MotionCtrl with VideoCrafter and CameraCt r1 with AnimateDiffV3 as base model, respectively. The last two rows are MotionCtrl and CameraCt r1 with SVD as base model taking the image as a condition signal. Condition images are the first images of each row.](images/3.jpg)
*该图像是一个比较示意图，展示了CameraCtrl与MotionCtrl在不同视频生成模型中的效果对比。上两行为T2V设置，分别展示了使用VideoCrafter的MotionCtrl与使用AnimateDiffV3的CameraCtrl。下两行为以图像作为条件信号的比较，包含了条件图像的信息。*

*   **First two rows (T2V setting):** `MotionCtrl` (first row) fails to follow the camera condition, exhibiting scene rotation rather than camera movement. In contrast, `CameraCtrl` (second row) accurately distinguishes and follows the camera trajectory condition, showing clear camera motion around the subject (e.g., the horse).
*   **Last two rows (I2V setting):** `MotionCtrl` (third row) is insensitive to small camera movements, primarily showing forward motion while ignoring a slight leftward movement in the condition. `CameraCtrl` (fourth row) precisely follows both forward and subtle leftward movements, demonstrating finer control.
*   **Conclusion:** Qualitative results reinforce the quantitative findings, showing `CameraCtrl`'s superior ability to precisely follow complex and subtle camera trajectories compared to `MotionCtrl`.

## 6.2. Ablation Studies / Parameter Analysis

The paper conducts extensive ablation studies to justify design choices regarding camera representation, camera encoder architecture, and camera feature injection points.

### 6.2.1. Camera Representation (Table 2a)
The following are the results from Table 2a of the original paper:

<table><tr><td colspan="3">Representation type FVD↓TransErr↓RotErr↓</td></tr><tr><td>Raw Values</td><td>230.1</td><td>13.88</td><td>1.51</td></tr><tr><td>Euler angles</td><td>221.2</td><td>13.71</td><td>1.43</td></tr><tr><td>Direction + Origin</td><td>232.3</td><td>13.21</td><td>1.57</td></tr><tr><td>Plücker embedding</td><td>222.1</td><td>12.98</td><td>1.29</td></tr></table>

*   **Raw Values:** Directly using intrinsic K and extrinsic E matrices.
*   **Euler angles:** Converting rotation matrix to Euler angles, then repeating spatially.
*   **Direction + Origin:** Combining ray directions (pixel-varying) and a repeated camera origin (constant across spatial positions).
*   **Plücker embedding:** The proposed method.

**Analysis:**
*   `Plücker embedding` yields the best camera control results (`TransErr` 12.98, `RotErr` 1.29) with comparable FVD.
*   **Why:** `Plücker embedding` provides a `geometric interpretation for every pixel`, allowing better correlation between camera pose and visual details. Raw numerical values or Euler angles suffer from numerical mismatches or lack pixel-wise information, hindering learning. The `Direction + Origin` method introduces redundancy with repeated camera origin parameters, potentially misaligning features.
*   **Conclusion:** `Plücker embedding` is the most effective camera representation.

### 6.2.2. Camera Encoder Architecture (Table 2b)
The following are the results from Table 2b of the original paper:

<table><tr><td colspan="4">Encoder architecture typeFVD↓TransErr ↓RotErr ↓</td></tr><tr><td>ControlNet</td><td>295.8</td><td>13.51</td><td>1.42</td></tr><tr><td>ControlNet + Temporal</td><td>283.4</td><td>13.13</td><td>1.33</td></tr><tr><td>T2I Adaptor</td><td>223.4</td><td>13.27</td><td>1.38</td></tr><tr><td>T2I Adaptor + Temporal</td><td>222.1</td><td>12.98</td><td>1.29</td></tr></table>

*   **ControlNet:** Encoder that takes sum of image features and Plücker embedding.
*   **ControlNet + Temporal:** `ControlNet` with added temporal attention blocks.
*   **T2I Adaptor:** Encoder that takes only Plücker embedding as input.
*   **T2I Adaptor + Temporal:** `T2I Adaptor` with added temporal attention blocks (the chosen design for `CameraCtrl`).

**Analysis:**
*   `T2I Adaptor` variants (`T2I Adaptor` and `T2I Adaptor + Temporal`) achieve significantly better FVD scores (around 220-223) compared to `ControlNet` variants (283-295), indicating better appearance quality and less `appearance leakage`. This validates the choice of taking only `Plücker embedding` as input to the camera encoder, making it agnostic to appearance bias.
*   Adding `Temporal attention` modules to the encoder (`T2I Adaptor + Temporal` vs `T2I Adaptor`, and `ControlNet + Temporal` vs `ControlNet`) consistently improves camera control (lower `TransErr` and `RotErr`).
*   **Conclusion:** The `T2I Adaptor` encoder enhanced with `temporal attention modules` is the optimal camera encoder architecture, providing both good appearance quality and precise camera control.

### 6.2.3. Injection Place for Camera Features (Table 2c)
The following are the results from Table 2c of the original paper:

<table><tr><td>Attention</td><td>FVD↓TransErr↓RotErr↓</td><td></td></tr><tr><td>Spatial Self</td><td>241.2</td><td>14.72</td><td>1.42</td></tr><tr><td>Spatial Cross</td><td>237.5</td><td>14.31</td><td>1.51</td></tr><tr><td>Spatial Self + Cross</td><td>240.1</td><td>14.52</td><td>1.60</td></tr><tr><td>Temporal</td><td>222.1</td><td>12.98</td><td>1.29</td></tr></table>

**Analysis:**
*   Injecting camera features into the `Temporal attention` layers yields the best camera control metrics (`TransErr` 12.98, `RotErr` 1.29) and competitive FVD.
*   Injecting into `Spatial Self`, `Spatial Cross`, or `Both` spatial attention layers results in higher errors.
*   **Why:** Camera motion induces `global view changes across frames`, which is inherently a temporal phenomenon. Integrating camera poses with the `temporal blocks` of the `U-Net` (which capture inter-frame relationships) aligns well with this dynamic nature, leading to superior control.
*   **Conclusion:** Camera features should be injected into the `temporal attention layers` of the `U-Net`.

### 6.2.4. Effect of Datasets (Table 2d)
The following are the results from Table 2d of the original paper:

<table><tr><td colspan="4">Datasets FVD↓TransErr ↓RotErr↓</td></tr><tr><td>Objaverse</td><td>1435.4</td><td>Incapable</td><td>Incapable</td></tr><tr><td>MVImageNet</td><td>1143.5</td><td>113.87</td><td>1.52</td></tr><tr><td>RealEstate10K + ACID</td><td>1102.4</td><td>13.48</td><td>1.41</td></tr><tr><td>RealEstate10K</td><td>1088.9</td><td>12.99</td><td>1.39</td></tr></table>

**Analysis:**
*   **Objaverse:** Produces a very high FVD (1435.4) and `COLMAP` struggles to extract meaningful camera poses ("Incapable" for errors). This confirms the hypothesis that the `appearance distribution gap` (synthetic data vs. real-world base models) significantly hinders performance and generalization.
*   **MVImageNet:** While better than `Objaverse`, it still shows a high FVD (1143.5) and very high `TransErr` (113.87) compared to `RealEstate10K`. This is attributed to its `lack of diverse camera trajectories` (mostly horizontal rotations), leading to poor generalization.
*   **RealEstate10K:** Achieves the best performance (FVD 1088.9, `TransErr` 12.99, `RotErr` 1.39), demonstrating the optimal trade-off.
*   **RealEstate10K + ACID:** Combining `RealEstate10K` with another similar but smaller dataset (`ACID`) did not improve results, suggesting that `RealEstate10K` already covers a good range of camera diversity, and simply adding more data without significantly increasing camera distribution complexity is not beneficial.
*   **Conclusion:** Training on datasets with `diverse camera distributions` and `similar appearance to the base model's training data` (like `RealEstate10K`) is crucial for achieving high controllability and generalization. The current bottleneck for further improvement might be the complexity of camera pose distribution in available datasets.

### 6.2.5. Lower Bound of TransErr and RotErr (Table 5 in Appendix F.3)
The following are the results from Table 5 of the original paper:

<table><tr><td></td><td>TransErr↓</td><td>RotErr↓</td></tr><tr><td>Lower Bounds</td><td>6.93</td><td>1.02</td></tr></table>

**Analysis:**
*   This table provides the estimated lower bounds for `TransErr` and `RotErr` on the `RealEstate10K` test set, obtained by running `COLMAP` on ground truth video clips and comparing its output against the actual ground truth.
*   These values (6.93 for `TransErr` and 1.02 for `RotErr`) represent the inherent error margin due to `COLMAP`'s instability and limitations. `CameraCtrl`'s results (e.g., `TransErr` 9.02, `RotErr` 1.18 for `CameraCtrlSVD`) are relatively close to these lower bounds, indicating high accuracy given the challenges of camera pose extraction.

## 6.3. Applications of CameraCtrl (Figure 4)

The following figure (Figure 4 from the original paper) showcases the applications of CameraCtrl:

![Figure 4: Applications of CameraCtrl. The first row represents a video generated by the base AnimateDiff. The Following two rows showcase the results of two personalized T2V generators, RealisticVision and ToonYou. The fourth row expresses the video generated by CameraCtrl integrated with another video control method, SparseCtrl (Guo et al., 2023a). The video of the last row is produced by a I2V generator, SVD, taking the first image of last row as a condition.](images/4.jpg)
*该图像是展示CameraCtrl应用的示意图。第一行展示了由基础AnimateDiff生成的视频，第二、三行分别展示了两个个性化的文本生成视频结果，分别为RealisticVision和ToonYou。第四行展示了CameraCtrl与另一种视频控制方法SparseCtrl结合生成的结果，最后一行是由I2V生成器SVD产生的视频，以倒数第一行的第一张图像作为条件。*

`CameraCtrl`'s design (using `Plücker embeddings` and a `plug-and-play` module agnostic to appearance) enables broad applicability across various video generators and compatibility with other control methods.

*   **Diverse T2V Generators:**
    *   **First Row (Base `AnimateDiff`):** Shows a video of a natural scene generated by the base `AnimateDiff` with `CameraCtrl` applied.
    *   **Second Row (Personalized T2V - `Realistic Vision`):** Demonstrates `CameraCtrl` controlling the camera for a stylized video of a cyberpunk city, generated with `AnimateDiff` integrated with the `RealisticVision` personalized image generator.
    *   **Third Row (Personalized T2V - `ToonYou`):** Shows `CameraCtrl` controlling a video of a cartoon character, generated with `AnimateDiff` integrated with the `ToonYou` personalized image generator.
    *   **Analysis:** In all these diverse domains, `CameraCtrl` consistently demonstrates effective control over camera trajectories, highlighting its generalizability across different styles and content.

*   **I2V Generation (`SVD`):**
    *   **Last Row (I2V with `SVD`):** Illustrates `CameraCtrl` integrated with `Stable Video Diffusion` to generate a video from an input image, with controlled camera movement.
    *   **Analysis:** This shows `CameraCtrl`'s versatility in the `Image-to-Video` setting, extending its utility beyond text-based generation.

*   **Integration with Other Video Control Methods (`SparseCtrl`):**
    *   **Fourth Row (Integrated with `SparseCtrl`):** Displays a video generated by `CameraCtrl` combined with `SparseCtrl (Guo et al., 2023a)`. `SparseCtrl` manipulates specific frames (e.g., via RGB images or sketch maps) to control overall video generation. Here, the `RGB encoder` of `SparseCtrl` is used.
    *   **Analysis:** This demonstrates `CameraCtrl`'s `plug-and-play` nature and its ability to collaborate with other control modules. The generated video maintains high consistency with the reference image (from `SparseCtrl`) while accurately following the provided camera trajectory (from `CameraCtrl`). This opens possibilities for multi-modal, fine-grained control over both content and viewpoint.

**Flexibility of CameraCtrl (Appendix H.4):**
*   **Different Camera Movement Intensity (Figure 18):** `CameraCtrl` can control the intensity of camera movement by adjusting the interval between translation vectors of adjacent camera poses, allowing for more intense or gradual movements.
*   **Controlling Camera Movement by Adjusting Intrinsic (Figure 19):** By modifying the camera's intrinsic parameters (e.g., principal point `(cx, cy)` for translation, or focal length `(fx, fy)` for zoom-in/out), `CameraCtrl` can achieve various camera movements. This is possible because `Plücker embeddings` incorporate intrinsic parameters during computation.

## 6.4. Failure Cases (Figure 20)

The following figure (Figure 20 from the original paper) shows some failure cases of CameraCtrl:

![该图像是一个示意图，展示了通过精确的相机控制生成视频的过程。从左到右依次是相机的轨迹控制图以及不同相机角度生成的场景，体现出在文本与相机姿态输入下的动态视频生成能力。](images/20.jpg)
*该图像是一个示意图，展示了通过精确的相机控制生成视频的过程。从左到右依次是相机的轨迹控制图以及不同相机角度生成的场景，体现出在文本与相机姿态输入下的动态视频生成能力。*

*   **Problem:** `CameraCtrl` struggles when the desired camera rotation is of a large extent (e.g., 100 degrees vertical rotation in rows 1 and 2, or 150 degrees horizontal rotation in rows 3 and 4). The generated videos cannot reproduce the full extent of the requested rotation, only achieving a partial rotation (e.g., about 90 degrees for 150 degrees horizontal rotation).
*   **Reason:** The main reason identified is that the training dataset (`RealEstate10K`) may not contain enough camera trajectories with such large degrees of rotation.
*   **Future Work Implication:** This suggests that to further improve camera trajectory performance, a dataset with a larger and more extreme camera pose distribution, while still maintaining similar visual appearance to `RealEstate10K`, is needed.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary

The paper successfully introduces `CameraCtrl`, a novel and highly effective method for enabling precise camera pose control in video diffusion models. Addressing a critical gap in existing video generation capabilities, `CameraCtrl` offers users unprecedented control over cinematic viewpoints. The core innovations lie in its use of `Plücker embeddings` for a geometrically rich camera representation, a `plug-and-play` camera control module that integrates seamlessly without modifying base models, and a meticulously studied data-driven training strategy. Experimental results rigorously demonstrate `CameraCtrl`'s superior accuracy in camera control (lower `TransErr` and `RotErr`, higher `User Preference Rate`) compared to baselines like `MotionCtrl` and `AnimateDiff`, all while preserving or enhancing the visual quality and dynamism of generated videos. `CameraCtrl`'s broad applicability across diverse T2V, personalized T2V, and I2V settings, along with its compatibility with other control methods, marks a significant step forward in customized and dynamic video storytelling.

## 7.2. Limitations & Future Work

**Limitations identified by the authors:**
1.  **Limited Large-Rotation Handling:** `CameraCtrl` struggles to accurately reproduce camera trajectories involving very large rotations (e.g., 100+ degrees). This is attributed to the current training dataset (`RealEstate10K`) potentially lacking sufficient examples of such extreme camera movements.

**Future Research Directions suggested by the authors:**
1.  **Larger and More Diverse Camera Distribution Datasets:** To overcome the limitation of large rotations and further improve camera control accuracy, there is a need for new datasets that possess a similar visual appearance to `RealEstate10K` but contain a significantly larger and more complex camera pose distribution.
2.  **Broader Applications:** The plug-and-play nature of `CameraCtrl` positions it as a foundational advancement for related fields such as 3D and 4D content generation.
3.  **Ethical Oversight and Deepfake Detection:** The enhanced realism and controllability offered by `CameraCtrl` raise ethical concerns regarding privacy and the potential for creating misleading content (deepfakes). There is a critical need for ethical oversight and the development of more advanced deepfake detectors to manage these risks and ensure responsible usage.

## 7.3. Personal Insights & Critique

This paper presents a highly impactful contribution to the field of controllable video generation. The focus on camera control, often overlooked in favor of content control (e.g., object motion, scene structure), addresses a fundamental aspect of visual storytelling.

**Inspirations & Transferability:**
*   **Geometric Representation Power:** The choice of `Plücker embeddings` is a key insight. It highlights that carefully selected, geometrically informed representations can significantly outperform raw numerical parameters for complex 3D control tasks. This principle could be transferred to other domains requiring precise 3D manipulation, such as robot manipulation, scene editing, or even neural rendering. For instance, using similar pixel-wise geometric embeddings for light source control or object deformation could yield richer results.
*   **Plug-and-Play Modularity:** The `plug-and-play` design is brilliant. It allows for rapid integration with new base models and other control methods without requiring extensive re-training or modification of foundational models. This modularity is crucial for the rapid evolution of generative AI, where new base models emerge frequently. This design pattern could inspire more modular approaches in other complex AI systems, promoting composability and reusability.
*   **Data-Centric Design:** The rigorous ablation study on datasets reinforces the critical role of data characteristics. The finding that dataset appearance similarity to the base model's training data is as important as camera diversity is a practical guideline for future research in controllable generation. This insight can be applied to any task where a control module is trained on top of a pre-trained base model, emphasizing the importance of alignment in data distributions.

**Potential Issues, Unverified Assumptions, or Areas for Improvement:**
*   **Reliance on COLMAP for Metrics:** While `COLMAP` is a standard tool for `structure-from-motion`, the paper acknowledges its instability for short video clips and scale-invariance issues. The post-processing steps to address scale are good, but the inherent noise and potential inaccuracies of `COLMAP` as the primary evaluation tool for `TransErr` and `RotErr` might introduce a ceiling to reported accuracy. Developing or using more robust, perhaps learning-based, 3D metric estimation methods for generated videos could provide even more reliable evaluation.
*   **Computational Cost:** Training `CameraCtrl` requires substantial computational resources (16-32 A100 GPUs for 25-40 hours). While justified for research, this indicates that fine-grained camera control is still a computationally intensive task, which might be a barrier for broader adoption or iterative design for smaller research groups. Future work could explore more parameter-efficient ways to integrate camera control.
*   **Generalization to Novel Object Categories/Scenes:** While the paper shows generalization across different styles (e.g., `RealisticVision`, `ToonYou`), the `RealEstate10K` dataset primarily features architectural/indoor scenes. It would be interesting to see how `CameraCtrl` performs on completely novel object categories or highly dynamic natural scenes outside this domain, beyond just stylistic changes. The "appearance leakage" concern is well-addressed, but semantic differences in `content` might still pose challenges.
*   **Complexity of Plücker Embeddings for Users:** While powerful, `Plücker embeddings` are not intuitively understood by an average user. The `CameraCtrl` interface would likely abstract this away, allowing users to define trajectories via simpler keyframe animations or direct manipulation. However, the connection between user intent (e.g., "pan left slowly") and the underlying `Plücker embedding` is still a complex mapping that the model learns implicitly. Exploring more direct user-friendly interfaces that leverage these embeddings could be a rich area.
*   **Ethical Implications:** The authors thoughtfully included an ethics statement. The ability to precisely control camera viewpoints adds another layer of realism and potential for misuse. Proactive development of detection methods for synthetically generated content with controlled camera paths is crucial as this technology advances.

    Overall, `CameraCtrl` is a robust and well-designed system that pushes the boundaries of controllable video generation. Its principled approach to camera representation and modular design provides a strong foundation for future advancements in cinematic AI.