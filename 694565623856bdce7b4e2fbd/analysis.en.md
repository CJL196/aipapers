# 1. Bibliographic Information

## 1.1. Title
Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

The title clearly states the paper's core contribution: extending the concept of 3D Gaussians to handle dynamic (moving or changing) scenes. It specifies that the method is designed for high-fidelity reconstruction using input from a single camera (`monocular`) and involves making the 3D Gaussians `deformable` over time.

## 1.2. Authors
Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, Xiaogang Jin.

The authors are affiliated with the State Key Laboratory of CAD&CG at Zhejiang University and ByteDance Inc. This represents a collaboration between a leading academic research institution in computer graphics and a major technology company, suggesting a focus on both foundational research and practical, high-performance applications.

## 1.3. Journal/Conference
The paper was published on arXiv, an open-access repository for electronic preprints. This means it was shared publicly before or during a formal peer-review process for a conference or journal. Papers of this nature are typically submitted to top-tier computer graphics and computer vision conferences such as SIGGRAPH, CVPR, ICCV, or ECCV.

## 1.4. Publication Year
The paper was submitted to arXiv on September 22, 2023.

## 1.5. Abstract
The abstract introduces the problem that current state-of-the-art dynamic scene rendering methods, which heavily rely on implicit neural representations (like NeRF), often fail to capture fine details and cannot achieve real-time rendering speeds. To solve these issues, the authors propose a method based on **Deformable 3D Gaussian Splatting**. Their approach represents a scene using 3D Gaussians in a static "canonical space" and learns a deformation field to model their movement and changes over time from a monocular video. A key innovation is an **annealing smoothing training (AST) mechanism**, which improves temporal smoothness in real-world datasets with inaccurate camera poses, without adding computational overhead. The method leverages a differentiable Gaussian rasterizer to achieve both superior rendering quality and real-time performance. Experiments confirm that the proposed method significantly outperforms existing approaches in both quality and speed.

## 1.6. Original Source Link
*   **Original Source (arXiv):** https://arxiv.org/abs/2309.13101
*   **PDF Link:** https://arxiv.org/pdf/2309.13101v2.pdf
*   **Publication Status:** This is a preprint and has not yet been published in a peer-reviewed journal or conference proceeding at the time of its release.

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The reconstruction and rendering of dynamic scenes from video is a long-standing challenge in computer graphics and vision. The goal is to create a digital representation of a scene that can be viewed from novel viewpoints and at different points in time, even between captured frames.

*   **Core Problem:** Traditional methods like mesh-based representations often lack realism and struggle with complex topological changes. The advent of Neural Radiance Fields (NeRF) introduced implicit neural representations, which achieved photorealistic results but came with significant drawbacks:
    1.  **Lack of Detail:** Implicit representations can smooth over fine geometric and textural details.
    2.  **Slow Rendering:** The rendering process for NeRF-based methods requires querying a neural network hundreds of times per pixel, making real-time performance on dynamic scenes a major challenge.

*   **Existing Gaps:** While some methods have accelerated NeRF for static scenes, extending this speed to dynamic scenes is difficult. Dynamic scenes have higher complexity (or "rank," as the authors put it), which challenges the assumptions made by many acceleration techniques (e.g., grid-based methods).

*   **Innovative Idea:** The paper's key insight is to build upon the recently proposed **3D Gaussian Splatting (3D-GS)**. 3D-GS achieved unprecedented real-time rendering speeds and high-fidelity results for *static* scenes by using an explicit representation (millions of 3D Gaussians) and a highly optimized differentiable rasterizer. The authors' innovative step is to adapt this powerful static scene representation to the dynamic domain by introducing a **deformation field**. Instead of re-training a set of Gaussians for every frame, they learn a single set of Gaussians in a reference "canonical" space and a neural network that predicts how these Gaussians move, rotate, and scale over time.

## 2.2. Main Contributions / Findings
The paper makes several key contributions to the field of dynamic scene reconstruction:

1.  **A Deformable 3D-GS Framework:** This is the first work to extend the 3D Gaussian Splatting framework to model monocular dynamic scenes. By learning 3D Gaussians in a canonical space and deforming them over time, the method achieves both high-fidelity reconstruction and real-time rendering speeds, directly addressing the core limitations of previous methods.

2.  **Annealing Smoothing Training (AST):** The authors introduce a novel training mechanism to handle a practical but critical problem: inaccurate camera poses estimated from real-world videos. This inaccuracy often leads to jittery or inconsistent reconstructions. AST introduces decaying noise into the time input during training, which regularizes the deformation field and produces smoother temporal interpolations without any extra computational cost during inference.

3.  **State-of-the-Art Performance:** The proposed method is shown to significantly outperform existing dynamic neural rendering methods on both synthetic and real-world datasets. The improvements are not only in rendering speed but also in visual quality, particularly in capturing fine details and structural consistency, as measured by metrics like SSIM and LPIPS.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

*   **Neural Radiance Fields (NeRF):** NeRF is a method for synthesizing novel views of a complex 3D scene from a set of input images. It represents the scene using a fully-connected neural network (an MLP). This network takes a 5D coordinate as input—a 3D location `(x, y, z)` and a 2D viewing direction $(\theta, \phi)$—and outputs a volume density $\sigma$ (how opaque the point is) and an RGB color $\mathbf{c}$. To render an image, rays are cast from the camera through each pixel. The MLP is queried at multiple points along each ray, and the resulting colors and densities are combined using classical **volume rendering** principles to compute the final pixel color. While NeRF produces stunningly realistic images, this per-ray sampling process is computationally intensive, making rendering slow.

*   **3D Gaussian Splatting (3D-GS):** 3D-GS is a radical departure from the implicit representation of NeRF. It represents a scene explicitly as a large collection of 3D Gaussians. Each Gaussian is defined by several learnable parameters:
    *   **Position ($\mathbf{x}$):** Its 3D center.
    *   **Covariance ($\Sigma$):** A 3x3 matrix defining its shape and orientation (an ellipsoid). This is typically represented by a rotation (quaternion) and a scaling vector for easier optimization.
    *   **Opacity ($\sigma$):** How transparent it is.
    *   **Color (c):** Its appearance, represented by Spherical Harmonics (SH) to model view-dependent effects.
        To render an image, these 3D Gaussians are projected onto the 2D image plane, turning them into 2D Gaussians. These 2D Gaussians are then "splatted" onto the image and blended together in a front-to-back order to form the final pixel colors. The key innovation of 3D-GS is a custom, highly efficient, and **differentiable Gaussian rasterizer** implemented in CUDA, which allows for extremely fast rendering (over 100 FPS) and optimization.

*   **Canonical Space:** In the context of dynamic scenes, a canonical space is a shared, time-independent reference frame. The idea is to represent the geometry and appearance of an object or scene in a single, static configuration (the "canonical" form). A separate **deformation field** (often a neural network) is then trained to map points from this canonical space to their corresponding positions in the "live" scene at any given time $t$. This approach effectively **decouples** the static geometry from the dynamic motion, which is often a more efficient and generalizable way to model dynamic scenes.

## 3.2. Previous Works

*   **Entangled vs. Disentangled Dynamic NeRFs:**
    *   **Entangled Methods:** Early dynamic NeRF methods simply added time $t$ as a direct input to the NeRF MLP, i.e., $F(x, y, z, t, \theta, \phi) \rightarrow (\mathbf{c}, \sigma)$. This approach "entangles" motion and appearance, making it difficult for the network to learn consistent geometry over time and often requiring significant regularization.
    *   **Disentangled Methods (D-NeRF, Nerfies, HyperNeRF):** These methods, which are more relevant to the current paper, adopt the canonical space approach. For instance, **D-NeRF** uses two MLPs: a canonical NeRF that represents the scene at $t=0$, and a deformation MLP that takes a point `(x, y, z)` and time $t$ and outputs a displacement vector $\Delta \mathbf{x}$. A point is first warped from the observation space to the canonical space before its color and density are queried. **Nerfies** and **HyperNeRF** build on this idea to handle more complex, non-rigid deformations and even topological changes. However, all these methods inherit the slow rendering speed of NeRF.

*   **Accelerated Neural Rendering:** To combat NeRF's slowness, researchers developed hybrid methods that combine neural networks with explicit data structures like grids or planes.
    *   **Grid-based Methods (TensoRF, Instant-NGP):** These methods store features in an explicit voxel grid (or a decomposition of it). Instead of a large MLP, a small MLP is used to decode features sampled from the grid. **Instant-NGP** famously uses a multi-resolution hash grid to achieve very fast training.
    *   **Plane-based Methods (HexPlane, K-Planes):** These represent the 4D scene (3D space + 1D time) using a set of 2D planes, factorizing the 4D tensor into more compact representations.
    *   The authors of the current paper argue that while these methods are faster than pure MLP-based approaches, they rely on a low-rank assumption that may not hold for complex dynamic scenes, limiting their quality. Furthermore, they still rely on ray-casting, which is less efficient than the rasterization approach of 3D-GS.

## 3.3. Technological Evolution
The field has evolved in a clear trajectory:
1.  **NeRF (2020):** Established implicit neural representations for view synthesis, prioritizing quality over speed.
2.  **Dynamic NeRFs (2021+):** Extended NeRF to handle time-varying scenes, mostly by introducing deformation fields, but remained slow.
3.  **Accelerated NeRFs (2022+):** Introduced explicit data structures (grids, planes) to speed up training and rendering, making near-real-time performance possible for static scenes.
4.  **3D Gaussian Splatting (2023):** Shifted the paradigm from implicit (ray-casting) to explicit (rasterization), achieving true real-time rendering with state-of-the-art quality for static scenes.
5.  **Deformable 3D Gaussians (This Paper, 2023):** Represents the next logical step, adapting the highly successful explicit 3D-GS representation to the dynamic domain, aiming to bring its speed and quality benefits to video and time-varying content.

## 3.4. Differentiation Analysis
Compared to previous dynamic scene rendering methods, this paper's approach is fundamentally different in its choice of scene representation:

*   **Explicit Primitives vs. Implicit Fields:** While D-NeRF and its successors deform a continuous 3D coordinate space, this paper deforms a discrete set of explicit primitives (the 3D Gaussians). The scene's appearance and geometry are stored directly in the Gaussians' parameters, not in the weights of a large MLP.
*   **Rasterization vs. Ray-casting:** The rendering process is based on splatting/rasterization, not ray-casting. This avoids the costly step of sampling hundreds of points along each ray and allows the method to leverage the highly optimized GPU rasterization pipeline, which is the primary source of its speed advantage.
*   **Motion Modeling:** The deformation field learns to update all properties of the Gaussians (position, rotation, and scale), providing a richer model of motion compared to methods that only predict positional offsets.

    ---

# 4. Methodology

## 4.1. Principles
The core principle of the proposed method is to **decouple the static representation of a dynamic scene from its motion**. The scene's geometry and appearance are captured by a single set of 3D Gaussians in a time-independent **canonical space**. A separate, lightweight neural network, called the **deformation field**, is then trained to predict how each of these canonical Gaussians should be transformed (moved, rotated, and scaled) to reconstruct the scene at any given time $t$. The final image is rendered by applying these deformations and then feeding the transformed Gaussians into the standard 3D-GS differentiable rasterization pipeline.

The overall architecture is depicted in the paper's Figure 2.

![该图像是示意图，展示了可变形3D高斯用于动态场景重建的过程。图中包含了时间、初始化、3D高斯和图像生成等多个步骤，以及不同的流向与参数调整，体现了动态场景重建中的关键机制和流动关系。](images/2.jpg)
*该图像是示意图，展示了可变形3D高斯用于动态场景重建的过程。图中包含了时间、初始化、3D高斯和图像生成等多个步骤，以及不同的流向与参数调整，体现了动态场景重建中的关键机制和流动关系。*

## 4.2. Core Methodology In-depth

The method can be broken down into the following integrated steps.

### 4.2.1. Representation: 3D Gaussians in Canonical Space
The scene is initialized from a sparse point cloud (typically generated by Structure-from-Motion, SfM) as a set of 3D Gaussians, $G$. Each Gaussian is defined in a static, canonical reference frame by a set of learnable parameters:
*   Center position: $\mathbf{x} \in \mathbb{R}^3$
*   Rotation: represented by a quaternion $\mathbf{r} \in \mathbb{R}^4$
*   Scaling: a 3D vector $\mathbf{s} \in \mathbb{R}^3$
*   Opacity: a scalar $\sigma \in \mathbb{R}$
*   Appearance: represented by Spherical Harmonics (SH) coefficients.

    The 3D covariance matrix $\Sigma$ is constructed from the rotation and scaling components. A rotation matrix $R$ is derived from the quaternion $\mathbf{r}$, and a scaling matrix $S$ is created from the vector $\mathbf{s}$. The covariance is then:
$$
\Sigma = R S S ^ { T } R ^ { T }
$$
This factorization makes the optimization of the Gaussian's shape and orientation more stable. These parameters define the "base" state of the scene.

### 4.2.2. Motion Modeling: The Deformation Field
To bring the static canonical scene to life, a deformation network $\mathcal{F}_{\theta}$ is introduced. This network is an MLP whose goal is to predict the transformation for each Gaussian at a specific time $t$.

The input to the network is the canonical position $\mathbf{x}$ of a Gaussian and the time $t$. Crucially, both inputs are first passed through a positional encoding function $\gamma(\cdot)$ to help the network learn high-frequency functions. The network then outputs offsets for the Gaussian's position, rotation, and scale.
$$
( \delta \mathbf {x} , \delta \pmb { r } , \delta \pmb { s } ) = \mathcal { F } _ { \boldsymbol { \theta } } ( \gamma ( \mathrm { sg } ( \pmb { x } ) ) , \gamma ( t ) )
$$
Let's break down this formula:
*   $(\delta\mathbf{x}, \delta\mathbf{r}, \delta\mathbf{s})$ are the predicted offsets for the position, rotation quaternion, and scale vector, respectively.
*   $\mathcal{F}_{\theta}$ is the MLP with learnable weights $\theta$. The paper's appendix specifies its architecture: 8 fully connected layers with 256 hidden units and ReLU activations, with a skip connection.
*   $\mathrm{sg}(\mathbf{x})$ is the **stop-gradient** operation on the input position. This is a subtle but important detail. It means that during backpropagation, gradients from the loss function will flow back to update the MLP weights $\theta$ and the canonical Gaussian parameters $(\mathbf{r}, \mathbf{s}, \sigma, \text{SH})$, but the gradients will *not* flow from the deformation network's output back to its positional input $\mathbf{x}$. This prevents the deformation field from influencing the canonical positions, stabilizing training by keeping the two components' learning objectives separate.
*   $\gamma(\cdot)$ is the positional encoding function, defined as:
    $$
    \gamma ( p ) = ( \sin ( 2 ^ { k } \pi p ) , \cos ( 2 ^ { k } \pi p ) ) _ { k = 0 } ^ { L - 1 }
    $$
    This function maps a scalar input $p$ to a higher-dimensional vector, allowing the MLP to better capture fine details in space and time.

At any given time $t$, a canonical Gaussian $( \mathbf{x}, \mathbf{r}, \mathbf{s}, \sigma)$ is transformed into a "live" Gaussian $( \mathbf{x}+\delta\mathbf{x}, \mathbf{r}+\delta\mathbf{r}, \mathbf{s}+\delta\mathbf{s}, \sigma)$.

### 4.2.3. Rendering: Differentiable Gaussian Splatting
With the set of deformed Gaussians for a given time $t$, the rendering process follows the standard 3D-GS pipeline:
1.  **Projection:** Each 3D Gaussian is projected onto the 2D image plane using the camera's view matrix $V$ and the Jacobian $J$ of the projective transformation. This results in a 2D covariance matrix $\Sigma'$:
    $$
    \Sigma ^ { \prime } = J V \Sigma V ^ { T } J ^ { T }
    $$
2.  **Rasterization and Blending:** The projected 2D Gaussians are sorted by depth and blended in a front-to-back manner to compute the final color $C(\mathbf{p})$ for each pixel $\mathbf{p}$. The color is an accumulation of the colors $c_i$ of all Gaussians $i$ that overlap the pixel, weighted by their opacity $\alpha_i$ and the accumulated transmittance $T_i$:
    $$
    C ( { \bf p } ) = \sum _ { i \in N } T _ { i } \alpha _ { i } c _ { i }
    $$
    where the opacity $\alpha_i$ is calculated from the Gaussian's learned opacity $\sigma_i$ and its 2D distribution, and the transmittance $T_i$ is the product of the opacities of all Gaussians in front of it: $T_i = \prod_{j=1}^{i-1} (1-\alpha_j)$.

This entire rendering pipeline is differentiable, allowing gradients from the image loss to flow back to the parameters of the deformed Gaussians.

### 4.2.4. Joint Optimization and Adaptive Control
The model is trained by minimizing a loss function between the rendered image and the ground truth image. The paper uses a combination of L1 loss and D-SSIM (structural dissimilarity) loss:
$$
\mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { \mathrm { D - S S I M } }
$$
where $\lambda$ is a weighting factor (set to 0.2).

The gradients from this loss are used to jointly optimize both:
*   The parameters of the **canonical 3D Gaussians** $(\mathbf{x}, \mathbf{r}, \mathbf{s}, \sigma, \text{SH})$.
*   The weights $\theta$ of the **deformation network** $\mathcal{F}_{\theta}$.

    Additionally, the method adopts the **adaptive density control** mechanism from 3D-GS. Periodically during training, Gaussians that are nearly transparent are pruned. In regions with large positional gradients (indicating undersampling), large Gaussians are split into smaller ones, and small Gaussians are cloned to fill in details. This process dynamically refines the set of canonical Gaussians to better represent the scene's geometry.

### 4.2.5. Annealing Smooth Training (AST) for Real-World Data
A key challenge with real-world monocular videos is that camera poses estimated by SfM tools like COLMAP can be noisy, causing temporal jitter. To address this, the authors propose the **Annealing Smooth Training (AST)** mechanism.

Instead of feeding the exact time $t$ to the deformation network, they add a small amount of random noise that decays over the course of training:
$$
\Delta = \mathcal { F } _ { \theta } \left( \gamma ( \mathrm { s g } ( \pmb { x } ) ) , \gamma ( t ) + \mathcal { X } ( i ) \right)
$$
where the noise term $\mathcal{X}(i)$ at training iteration $i$ is defined as:
$$
\mathcal { X } ( i ) = \mathbb { N } ( 0 , 1 ) \cdot \beta \cdot \Delta t \cdot ( 1 - i / \tau )
$$
Here:
*   $\mathbb{N}(0, 1)$ is a random number from a standard Gaussian distribution.
*   $\beta$ is a scaling factor (0.1).
*   $\Delta t$ is the mean time interval between frames.
*   $\tau$ is the total number of iterations for the annealing schedule (e.g., 20k).

    At the beginning of training, this adds significant temporal noise, forcing the deformation network to learn a smoother, more generalized function of time. As training progresses ($i \rightarrow \tau$), the noise term $(1 - i/\tau)$ decays to zero, allowing the network to fine-tune its predictions on the precise, noise-free time values. This acts as a form of curriculum learning or regularization, improving temporal smoothness without adding any computational overhead at inference time.

---

# 5. Experimental Setup

## 5.1. Datasets
The method was evaluated on a combination of standard synthetic and real-world datasets for dynamic scene reconstruction.

*   **D-NeRF Dataset (Synthetic):** This dataset, introduced by the D-NeRF paper, consists of 8 synthetic scenes with complex non-rigid motion, such as a Lego bulldozer, a jumping person (`Jumping Jacks`), and a T-Rex. The camera moves around the dynamic object. All experiments were conducted on full-resolution ($800 \times 800$) images against a black background.
*   **HyperNeRF Dataset (Real-world):** This dataset features challenging real-world scenes captured with a handheld camera, exhibiting both camera and object motion, and even topological changes (e.g., a person cutting a lemon). The paper notes that the camera poses for this dataset are often very inaccurate.
*   **NeRF-DS Dataset (Real-world):** This dataset contains 7 real-world scenes of dynamic specular objects, providing another challenging testbed for real-world performance.

    The choice of these datasets allows for a comprehensive evaluation, testing the method's ability to handle clean, synthetic motion as well as the noisy conditions of real-world captures.

## 5.2. Evaluation Metrics
The performance of the method is quantified using three standard image quality metrics:

1.  **PSNR (Peak Signal-to-Noise Ratio):**
    *   **Conceptual Definition:** PSNR measures the ratio between the maximum possible power of a signal (the maximum pixel value) and the power of corrupting noise that affects the fidelity of its representation. In image quality assessment, it measures the pixel-wise difference between the ground truth and the rendered image. Higher PSNR values indicate better reconstruction quality.
    *   **Mathematical Formula:**
        $$
        \text{PSNR} = 20 \cdot \log_{10}(\text{MAX}_I) - 10 \cdot \log_{10}(\text{MSE})
        $$
    *   **Symbol Explanation:**
        *   $\text{MAX}_I$ is the maximum possible pixel value of the image (e.g., 255 for an 8-bit image).
        *   $\text{MSE}$ is the Mean Squared Error between the ground truth and predicted images.

2.  **SSIM (Structural Similarity Index Measure):**
    *   **Conceptual Definition:** SSIM is a perceptual metric that assesses image quality by comparing three key characteristics: luminance, contrast, and structure. It is designed to be more consistent with human visual perception than pixel-based metrics like PSNR. SSIM values range from -1 to 1, where 1 indicates a perfect match.
    *   **Mathematical Formula:** For two image windows $x$ and $y$:
        $$
        \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
        $$
    *   **Symbol Explanation:**
        *   $\mu_x, \mu_y$ are the average pixel values of windows $x$ and $y$.
        *   $\sigma_x^2, \sigma_y^2$ are the variances.
        *   $\sigma_{xy}$ is the covariance of $x$ and $y$.
        *   $c_1, c_2$ are small constants to stabilize the division.

3.  **LPIPS (Learned Perceptual Image Patch Similarity):**
    *   **Conceptual Definition:** LPIPS is a metric that aims to better capture human perception of image similarity by using features extracted from deep convolutional neural networks (like AlexNet or VGG). It computes the distance between the deep feature representations of two image patches. Lower LPIPS values indicate that the two images are more perceptually similar.
    *   **Mathematical Formula:**
        $$
        d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} || w_l \odot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l) ||_2^2
        $$
    *   **Symbol Explanation:**
        *   $d(x, x_0)$ is the distance between images $x$ and $x_0$.
        *   $\hat{y}^l, \hat{y}_0^l$ are the feature activations from layer $l$ of a deep network for each image.
        *   $w_l$ are channel-wise weights to scale the importance of activations.
        *   The distance is computed patch-wise and averaged over the image.

## 5.3. Baselines
The proposed method was compared against a strong set of state-of-the-art methods for dynamic scene rendering:
*   **3D-GS:** The original static 3D Gaussian Splatting method, applied here as a baseline to show the necessity of a dynamic model.
*   **D-NeRF:** A foundational method for dynamic scenes using a deformation field with a canonical NeRF.
*   **TiNeuVox:** A fast dynamic NeRF method that uses time-aware neural voxels.
*   **Tensor4D:** A method that uses 4D tensor decomposition to represent dynamic scenes efficiently.
*   **K-Planes:** A method that represents the 4D space-time scene using explicit feature planes.
*   **HyperNeRF** and **NeRF-DS:** State-of-the-art methods for handling complex real-world dynamic scenes, which were also the sources of the real-world datasets.

    These baselines cover a range of approaches, from classic deformation-based NeRFs to more recent, highly-optimized hybrid methods, providing a robust benchmark for comparison.

---

# 6. Results & Analysis

## 6.1. Core Results Analysis

The experimental results demonstrate the clear superiority of the proposed Deformable 3D Gaussians framework over existing methods in both rendering quality and speed.

### 6.1.1. Synthetic Dataset Results
The following are the results from Table 1 of the original paper, comparing methods on the D-NeRF synthetic dataset.

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Hell Warrior</th>
<th colspan="3">Mutant</th>
<th colspan="3">Hook</th>
<th colspan="3">Bouncing Balls</th>
</tr>
<tr>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
<th>PSNR↑</th>
<th>SSIM↑</th>
<th>LPIPS↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>3D-GS</td>
<td>29.89</td>
<td>0.9155</td>
<td>0.1056</td>
<td>24.53</td>
<td>0.9336</td>
<td>0.0580</td>
<td>21.71</td>
<td>0.8876</td>
<td>0.1034</td>
<td>23.20</td>
<td>0.9591</td>
<td>0.0600</td>
</tr>
<tr>
<td>D-NeRF</td>
<td>24.06</td>
<td>0.9440</td>
<td>0.0707</td>
<td>30.31</td>
<td>0.9672</td>
<td>0.0392</td>
<td>29.02</td>
<td>0.9595</td>
<td>0.0546</td>
<td>38.17</td>
<td>0.9891</td>
<td>0.0323</td>
</tr>
<tr>
<td>TiNeuVox</td>
<td>27.10</td>
<td>0.9638</td>
<td>0.0768</td>
<td>31.87</td>
<td>0.9607</td>
<td>0.0474</td>
<td>30.61</td>
<td>0.9599</td>
<td>0.0592</td>
<td>40.23</td>
<td>0.9926</td>
<td>0.0416</td>
</tr>
<tr>
<td>Tensor4D</td>
<td>31.26</td>
<td>0.9254</td>
<td>0.0735</td>
<td>29.11</td>
<td>0.9451</td>
<td>0.0601</td>
<td>28.63</td>
<td>0.9433</td>
<td>0.0636</td>
<td>24.47</td>
<td>0.9622</td>
<td>0.0437</td>
</tr>
<tr>
<td>K-Planes</td>
<td>24.58</td>
<td>0.9520</td>
<td>0.0824</td>
<td>32.50</td>
<td>0.9713</td>
<td>0.0362</td>
<td>28.12</td>
<td>0.9489</td>
<td>0.0662</td>
<td>40.05</td>
<td>0.9934</td>
<td>0.0322</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>41.54</b></td>
<td><b>0.9873</b></td>
<td><b>0.0234</b></td>
<td><b>42.63</b></td>
<td><b>0.9951</b></td>
<td><b>0.0052</b></td>
<td><b>37.42</b></td>
<td><b>0.9867</b></td>
<td><b>0.0144</b></td>
<td><b>41.01</b></td>
<td><b>0.9953</b></td>
<td><b>0.0093</b></td>
</tr>
</tbody>
</table>

*(Note: The full table in the paper contains 8 scenes; a representative subset is shown above for brevity and clarity of analysis.)*

*   **Analysis:** The proposed method (`Ours`) achieves a massive improvement across all metrics and all scenes. The PSNR gains are substantial (e.g., 41.54 vs. 31.26 on `Hell Warrior`), but the most telling improvements are in SSIM and LPIPS. For example, on the `Mutant` scene, the LPIPS score is 0.0052, which is an order of magnitude better than the next best competitor (K-Planes at 0.0362). This indicates that the rendered images are not just pixel-perfect but are also structurally and perceptually much closer to the ground truth. The qualitative results in Figure 3 visually confirm this, showing that the proposed method produces significantly sharper and more detailed renderings, free of the blurriness common in NeRF-based methods.

    ![该图像是插图，展示了不同方法在动态场景重建中的效果比较。每一行代表一个场景，包括“恐龙”、“站立”、“突变体”、“跳远”等，通过不同的方法如“GT”、“Ours”、“TiNeuVox”、“K-Planes”等进行重建。图中包含场景的高质量重建结果，并显示了每种方法在细节表现和渲染质量上的差异。](images/3.jpg)
    *该图像是插图，展示了不同方法在动态场景重建中的效果比较。每一行代表一个场景，包括“恐龙”、“站立”、“突变体”、“跳远”等，通过不同的方法如“GT”、“Ours”、“TiNeuVox”、“K-Planes”等进行重建。图中包含场景的高质量重建结果，并显示了每种方法在细节表现和渲染质量上的差异。*

### 6.1.2. Real-World Dataset Results
The following are the results from Table 2 of the original paper, showing the mean metrics across all seven scenes of the NeRF-DS dataset.

<table>
<thead>
<tr>
<th></th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>3D-GS</td>
<td>20.29</td>
<td>0.7816</td>
<td>0.2920</td>
</tr>
<tr>
<td>TiNeuVox</td>
<td>21.61</td>
<td>0.8234</td>
<td>0.2766</td>
</tr>
<tr>
<td>HyperNeRF</td>
<td>23.45</td>
<td>0.8488</td>
<td>0.1990</td>
</tr>
<tr>
<td>NeRF-DS</td>
<td>23.60</td>
<td>0.8494</td>
<td>0.1816</td>
</tr>
<tr>
<td>Ours (w/o AST)</td>
<td>23.97</td>
<td>0.8346</td>
<td>0.2037</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>24.11</b></td>
<td><b>0.8525</b></td>
<td><b>0.1769</b></td>
</tr>
</tbody>
</table>

*   **Analysis:** On the challenging real-world NeRF-DS dataset, the proposed method again achieves the best performance across all metrics. The qualitative results in Figure 5 show that the method produces clearer results with fewer artifacts compared to baselines. The most important comparison here is between `Ours` and `Ours (w/o AST)`. The full method with Annealing Smooth Training achieves better scores on all metrics, confirming the effectiveness of AST in handling the noisy poses present in real-world data.

    ![该图像是一个示意图，展示了不同方法在动态场景重建中的表现，包括 GT、Ours、TiNeuVox、HyperNeRF、NeRF-DS 和 3D-GS。每列展示了在不同测试场景（as、bell 和 sieve）下的重建效果，对比了各方法的准确性和渲染质量。](images/5.jpg)
    *该图像是一个示意图，展示了不同方法在动态场景重建中的表现，包括 GT、Ours、TiNeuVox、HyperNeRF、NeRF-DS 和 3D-GS。每列展示了在不同测试场景（as、bell 和 sieve）下的重建效果，对比了各方法的准确性和渲染质量。*

### 6.1.3. Rendering Efficiency
The following are the results from Table 4 of the original paper, detailing the rendering speed (Frames Per Second) and the number of Gaussians for various scenes.

<table>
<thead>
<tr>
<th colspan="3">D-NeRF Dataset</th>
<th colspan="3">NeRF-DS Dataset</th>
<th colspan="3">HyperNeRF Dataset</th>
</tr>
<tr>
<th>Scene</th>
<th>FPS</th>
<th>Num (k)</th>
<th>Scene</th>
<th>FPS</th>
<th>Num (k)</th>
<th>Scene</th>
<th>FPS</th>
<th>Num (k)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Lego</td>
<td>24</td>
<td>300</td>
<td>AS</td>
<td>48</td>
<td>185</td>
<td>Espresso</td>
<td>15</td>
<td>620</td>
</tr>
<tr>
<td>Jump</td>
<td>85</td>
<td>90</td>
<td>Basin</td>
<td>29</td>
<td>250</td>
<td>Americano</td>
<td>6</td>
<td>1300</td>
</tr>
<tr>
<td>Bouncing</td>
<td>38</td>
<td>170</td>
<td>Bell</td>
<td>18</td>
<td>400</td>
<td>Cookie</td>
<td>9</td>
<td>1080</td>
</tr>
<tr>
<td>T-Rex</td>
<td>30</td>
<td>220</td>
<td>Cup</td>
<td>35</td>
<td>200</td>
<td>Chicken</td>
<td>10</td>
<td>740</td>
</tr>
<tr>
<td>Mutant</td>
<td>40</td>
<td>170</td>
<td>Plate</td>
<td>31</td>
<td>230</td>
<td>Torchocolate</td>
<td>8</td>
<td>1030</td>
</tr>
<tr>
<td>Warrior</td>
<td>172</td>
<td>40</td>
<td>Press</td>
<td>48</td>
<td>185</td>
<td>Lemon</td>
<td>23</td>
<td>420</td>
</tr>
<tr>
<td>Standup</td>
<td>93</td>
<td>80</td>
<td>Sieve</td>
<td>35</td>
<td>200</td>
<td>Hand</td>
<td>6</td>
<td>1750</td>
</tr>
<tr>
<td>Hook</td>
<td>45</td>
<td>150</td>
<td></td>
<td></td>
<td></td>
<td>Printer</td>
<td>12</td>
<td>650</td>
</tr>
</tbody>
</table>

*   **Analysis:** The method achieves real-time rendering speeds (>30 FPS) for many scenes, particularly when the number of Gaussians is below ~250k. The FPS is directly correlated with the number of Gaussians. An interesting finding is that scenes from the HyperNeRF dataset, which are known to have inaccurate camera poses, result in a significantly larger number of Gaussians (>1 million in some cases). This is likely because the adaptive densification algorithm tries to compensate for pose errors by adding more geometry, which in turn reduces rendering speed.

## 6.2. Ablation Studies / Parameter Analysis

The authors conducted several ablation studies to validate their design choices.

*   **Annealing Smooth Training (AST):** As shown in Table 2, adding AST improves performance on all metrics for the real-world NeRF-DS dataset (e.g., PSNR from 23.97 to 24.11, LPIPS from 0.2037 to 0.1769). Figure 4 provides a visual comparison, showing that AST helps create smoother renderings for interpolated time steps and captures finer details by mitigating overfitting to noisy poses.

    ![该图像是图表，展示了三种不同方法在动态场景重建中的效果对比。左侧为使用6个采样点且未应用退火平滑训练（6pe w/o AST），中间为使用10个采样点但同样未应用退火平滑训练（10pe w/o AST），右侧为本文提出的方法（ours）。每种方法的细节表现有所不同，特别是在动态效果和清晰度方面，强调了提出方法的优势。](images/4.jpg)
    *该图像是图表，展示了三种不同方法在动态场景重建中的效果对比。左侧为使用6个采样点且未应用退火平滑训练（6pe w/o AST），中间为使用10个采样点但同样未应用退火平滑训练（10pe w/o AST），右侧为本文提出的方法（ours）。每种方法的细节表现有所不同，特别是在动态效果和清晰度方面，强调了提出方法的优势。*

*   **Deformation Field Outputs:** The appendix provides a detailed ablation (Table 7) on the components of the deformation field's output $(\delta\mathbf{x}, \delta\mathbf{r}, \delta\mathbf{s})$. The experiments show that predicting offsets for all three components (position, rotation, and scale) yields the best results. Removing the scale offset (`w/o δs`) or rotation offset (`w/o δr`) leads to a degradation in quality, and removing both (`w/o r&s`) results in the worst performance. This confirms that modeling the full deformation of the Gaussians is beneficial.

*   **Depth Visualization:** Figure 6 shows depth maps rendered by the model. The clean and accurate depth maps indicate that the model is learning a geometrically correct representation of the scene, rather than "cheating" by just memorizing colors. This strong geometric foundation is crucial for achieving high-quality novel-view synthesis.

    ![Figure 6. Depth Visualization. We visualized the depth map of the D-NeRF dataset. The first row includes bouncing-balls, hellwarrior, hook, and jumping-jacks, while the second row includes lego, mutant, standup, and trex.](images/6.jpg)
    *该图像是深度可视化图表，展示了D-NeRF数据集中的深度图。第一行包括反弹球、地狱战士、钩子和跳跃者，第二行则包括乐高、突变体、站立和霸王龙。*

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper introduces a novel and highly effective method for monocular dynamic scene reconstruction called **Deformable 3D Gaussian Splatting**. By representing a scene with 3D Gaussians in a canonical space and learning a deformation field to animate them, the method successfully combines the high-fidelity representation of 3D-GS with the ability to model complex motion. This approach surpasses existing methods, which are typically based on slow implicit representations, in both rendering quality and speed, achieving real-time performance. Furthermore, the proposed **Annealing Smooth Training (AST)** mechanism provides a practical and zero-overhead solution to the common problem of noisy camera poses in real-world data, enhancing temporal smoothness while preserving detail.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations that point to avenues for future research:

*   **Dependence on Viewpoint Diversity:** The method's performance is sensitive to the number and variety of viewpoints in the input video. Sparse viewpoints can lead to overfitting, similar to other reconstruction methods.
*   **Sensitivity to Pose Accuracy:** While AST helps mitigate the effects of noisy poses, highly inaccurate poses can still cause the model to fail to converge or produce a large number of artifacts (and Gaussians), as seen with the HyperNeRF dataset.
*   **Scalability:** The training time and memory consumption are directly proportional to the number of Gaussians. Very complex scenes requiring millions of Gaussians can become computationally expensive.
*   **Limited Motion Complexity:** The method was primarily evaluated on scenes with moderate body or object motion. Its ability to handle highly intricate and subtle motions, such as nuanced facial expressions, has not yet been demonstrated.

## 7.3. Personal Insights & Critique
This paper is an excellent example of building on a breakthrough idea (3D-GS) and extending it logically to a new domain (dynamic scenes).

*   **Strengths:**
    *   **High Impact:** The work effectively solves the dual problem of speed and quality that plagued previous dynamic neural rendering methods. Achieving real-time, high-fidelity rendering of dynamic scenes is a significant step forward for applications like VR/AR and virtual production.
    *   **Elegant Formulation:** The concept of a canonical set of Gaussians deformed by an MLP is intuitive and powerful. It cleanly separates the scene's static properties from its dynamic behavior.
    *   **Practicality:** The AST mechanism is a simple yet clever solution to a real-world problem. It demonstrates an awareness of the practical challenges of data capture beyond idealized synthetic datasets.

*   **Potential Issues and Areas for Improvement:**
    *   **Nature of the Canonical Space:** The canonical space is learned implicitly and is not guaranteed to correspond to a meaningful "base pose" of the object. For articulated objects like humans, a more structured canonical representation (e.g., a T-pose) and deformation model (e.g., based on skinning) might yield better generalization.
    *   **Handling Topology Changes:** While the paper mentions it can handle some topological shifts, the current framework might struggle with extreme cases (e.g., water splashing, smoke) where the concept of a consistent set of deforming Gaussians breaks down. A mechanism to create and destroy Gaussians over time, not just during an initial refinement phase, could be a valuable extension.
    *   **The "Rank" Argument:** The authors argue against grid-based structures by stating that dynamic scenes have a higher "rank" than static ones. While intuitively true (time adds a dimension of complexity), this argument could be further formalized and explored. It's possible that a more sophisticated hybrid representation could still be beneficial.
    *   **Editing and Control:** The current representation is learned end-to-end for reconstruction. A major future direction would be to make the representation editable, allowing users to manipulate the motion or appearance of the scene after it has been captured, which the explicit nature of Gaussians is well-suited for.