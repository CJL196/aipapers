# 1. Bibliographic Information
## 1.1. Title
Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction

The title clearly indicates the paper's core contribution: a new method called `Momentum-GS` that uses **momentum-based self-distillation** with **3D Gaussian Splatting** to reconstruct **large-scale scenes** with high quality.

## 1.2. Authors
The authors are Jixuan Fan, Wanhua Li, Yifei Han, Tianru Dai, and Yansong Tang. Their affiliations are with Tsinghua University and Harvard University, which are top-tier research institutions. Wanhua Li and Yansong Tang, in particular, have published extensively in the fields of 3D vision, neural rendering, and computer graphics, lending credibility to the work.

## 1.3. Journal/Conference
The paper is available as a preprint on arXiv. The provided metadata indicates a submission date of December 6, 2024, suggesting it is slated for a future publication cycle. Given the authors' affiliations and the quality of the work, it is likely intended for a top-tier computer vision or graphics conference such as CVPR, ICCV, ECCV, or SIGGRAPH.

## 1.4. Publication Year
2024

## 1.5. Abstract
The abstract summarizes the key challenges and solutions of the paper. It highlights that while 3D Gaussian Splatting is effective for large-scene reconstruction, it suffers from high memory consumption. Hybrid representations (combining implicit and explicit features) can help, but when used with block-wise parallel training, they introduce two new problems: 1) training blocks independently reduces data diversity, hurting accuracy, and 2) the number of blocks is limited by the number of available GPUs.

To solve this, the authors propose `Momentum-GS`. The core idea is a **momentum-based self-distillation** framework. A "teacher" Gaussian decoder, which is updated slowly via momentum, provides a stable global reference. This teacher guides the training of each individual block, promoting consistency. Furthermore, they introduce **block weighting**, which dynamically adjusts each block's importance during training based on its reconstruction accuracy. The abstract claims that this method significantly outperforms existing techniques, achieving an 18.7% improvement in the LPIPS metric over `CityGaussian` and establishing a new state-of-the-art.

## 1.6. Original Source Link
-   **Original Source Link:** https://arxiv.org/abs/2412.04887
-   **PDF Link:** https://arxiv.org/pdf/2412.04887v2.pdf
-   **Publication Status:** This is a preprint available on arXiv. It has not yet been peer-reviewed or officially published in a conference or journal.

# 2. Executive Summary
## 2.1. Background & Motivation
The core problem addressed by this paper is the **efficient and consistent reconstruction of large-scale 3D scenes**. Modern techniques like 3D Gaussian Splatting (`3D-GS`) can produce stunningly realistic renderings but face significant challenges when scaled up to city-sized environments.

**Key Challenges in Prior Research:**
1.  **Memory & Storage Overload:** `3D-GS` represents a scene using millions or even billions of explicit 3D Gaussians. Storing and training these is extremely demanding on memory (VRAM) and disk space.
2.  **The "Divide-and-Conquer" Dilemma:** A common strategy to handle large scenes is to partition them into smaller, manageable blocks and process them in parallel. However, this introduces a new set of problems:
    *   **Inconsistency at Boundaries:** If each block is trained completely independently (as in methods like `CityGaussian`), there is no information shared between them. This often leads to visible seams or artifacts at block boundaries, such as sudden changes in lighting or color, as shown in Figure 1.
    *   **Scalability Bottleneck:** To improve consistency, blocks can be trained in parallel while sharing a common model component (like a decoder). However, this approach physically tethers the number of blocks you can process simultaneously to the number of GPUs you have, severely limiting scalability.
    *   **Hybrid Representation Issues:** To combat the memory issue, `hybrid representations` (e.g., `Scaffold-GS`) were proposed. They use a small neural network (an implicit decoder) to generate the properties of Gaussians on the fly, reducing storage. But when applied to the divide-and-conquer strategy, they face the same dilemma: independent decoders for each block can't be merged, while a shared decoder is limited by the GPU count (as illustrated in Figure 2).

        **Paper's Entry Point:** The authors identify the central tension between **scalability** (needing many blocks) and **consistency** (needing information sharing between blocks). Their innovative idea is to adapt a **teacher-student self-distillation framework** to this problem. This allows them to decouple the number of blocks from the GPU count while using the "teacher" model as a stable, global source of truth to enforce consistency across all blocks, even if they are trained at different times.

The following figure from the paper illustrates the problem with independent training and the limitation of simple parallel training, contrasting them with the proposed momentum-based approach.

![Figure 2. Comparison of three approaches for using hybrid representations to reconstruct large-scale scenes in a divideand-conquer manner. Examples with two blocks: (a) Independent training of each block, resulting in separate models that cannot be merged due to independent Gaussian Decoders, complicating rendering; (b) Parallel training with a shared Gaussian decoder, allowing merged output but limited by GPU count; (c) Our approach with a Momentum Gaussian Decoder, providing global guidance to each block and improving consistency across blocks.](images/2.jpg)
*该图像是示意图，展示了三种使用混合表示法重建大规模场景的方法对比，包括独立训练（a）、并行训练（b）和我们的动量自蒸馏训练（c）。其中通过动量教师高斯解码器为每个块提供全局指导，有助于提高块之间的一致性和重建准确性。*

## 2.2. Main Contributions / Findings
The paper presents three primary contributions to solve the aforementioned problems:

1.  **Scene Momentum Self-Distillation:** The authors introduce a teacher-student learning paradigm. A "student" Gaussian decoder is shared across all blocks and trained normally. A "teacher" decoder is not trained via backpropagation but is instead a slow-moving average of the student's weights (updated with momentum). This teacher provides a stable, scene-wide reference, and a consistency loss forces the student to align with it. This mechanism crucially **decouples the number of blocks from the number of GPUs**, allowing for massively scalable training.
2.  **Reconstruction-guided Block Weighting:** To further improve consistency, the paper proposes a dynamic weighting scheme. During training, the system tracks the reconstruction quality (PSNR and SSIM) of each block. Blocks that are performing poorly (i.e., have higher error) are given a higher weight in the overall loss function. This forces the shared decoder to prioritize improving the weaker parts of the scene, preventing it from overfitting to easy regions and promoting uniform quality.
3.  **State-of-the-Art Performance with High Efficiency:** `Momentum-GS` is shown to achieve superior reconstruction quality, particularly in perceptual metrics like LPIPS, compared to previous state-of-the-art methods. It does so while using significantly fewer blocks, less storage, and in some cases, less memory, demonstrating the strong potential of hybrid representations when combined with their novel training strategy.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
### 3.1.1. 3D Gaussian Splatting (3D-GS)
`3D Gaussian Splatting` is a rasterization-based method for novel view synthesis and 3D scene representation. Instead of using a continuous volumetric representation like NeRF, it models a scene as a collection of millions of 3D Gaussians.

Each Gaussian is defined by several key properties:
*   **Position ($\mu$):** A 3D coordinate representing the center of the Gaussian.
*   **Covariance ($\Sigma$):** A 3x3 matrix that defines the shape and orientation (ellipsoid) of the Gaussian. For efficiency, this is often represented by a 3D scaling vector and a quaternion for rotation.
*   **Color ($c$):** The color of the Gaussian, typically represented by `Spherical Harmonics (SH)` to model view-dependent effects (i.e., how the color changes depending on the viewing direction).
*   **Opacity ($\alpha$):** A scalar value representing the transparency of the Gaussian.

    **Rendering Process:** To render an image from a specific viewpoint, the 3D Gaussians are projected onto the 2D image plane. This projection transforms each 3D Gaussian into a 2D Gaussian "splat". These splats are then blended together in depth order (from back to front) to compute the final color for each pixel. This process, known as `alpha blending`, is described by the formula:
\$
C ( \mathbf { x } ^ { \prime } ) = \sum _ { i \in N } c _ { i } \sigma _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } )
\$
where $C(\mathbf{x}')$ is the final pixel color, $c_i$ is the color of the $i$-th Gaussian, and $\sigma_i$ is its contribution (based on its opacity and 2D Gaussian value at the pixel). This rasterization approach is highly parallelizable and significantly faster than the ray marching used in NeRFs.

### 3.1.2. Neural Radiance Fields (NeRF)
`NeRF` is a pioneering method that represents a 3D scene using a fully connected neural network (an `MLP`). This network takes a 3D coordinate `(x, y, z)` and a 2D viewing direction $(\theta, \phi)$ as input and outputs the color and volume density at that point. To render an image, rays are cast from the camera through each pixel. The network is queried at multiple points along each ray, and the resulting colors and densities are integrated using classical volume rendering principles to compute the final pixel color. While NeRF produces highly realistic images, this dense sampling process along each ray makes both training and rendering very slow.

### 3.1.3. Hybrid Representations
`Hybrid representations` aim to get the best of both explicit and implicit worlds. They combine an explicit data structure (like a voxel grid, planes, or anchor points) with an implicit neural network. This allows them to be more memory-efficient than purely explicit methods (like storing billions of Gaussians) and faster than purely implicit methods (like NeRF). `Scaffold-GS`, mentioned in the paper, is a prime example. It uses a set of explicit `anchor points` in 3D space, each with a learned feature vector. A small MLP (the decoder) then takes an anchor's feature and viewing information to generate the full properties of a Gaussian on-the-fly. This means only the anchor features and the small MLP need to be stored, significantly reducing the model size.

### 3.1.4. Self-Distillation and Momentum Update
`Self-distillation` is a training technique where a model learns from itself. In a teacher-student framework, a "student" model is trained to match the output of a "teacher" model, in addition to learning from the ground-truth data.

The key innovation in methods like `Momentum Contrast (MoCo)` is how the teacher is updated. Instead of being a separate, fixed model or being trained with backpropagation, the teacher is an **exponential moving average (EMA)** of the student model. Its parameters ($\theta_t$) are updated based on the student's parameters ($\theta_s$) using a momentum coefficient ($m$):
\$
\theta _ { t } \leftarrow m \cdot \theta _ { t } + ( 1 - m ) \cdot \theta _ { s }
\$
When $m$ is high (e.g., 0.999), the teacher updates very slowly. This makes it a more stable and consistent target for the student to learn from, which is especially useful in scenarios with noisy or rapidly changing training signals. `Momentum-GS` cleverly applies this concept to enforce consistency across different blocks of a large scene.

## 3.2. Previous Works
The paper positions itself relative to two main lines of research: large-scale NeRFs and large-scale 3D-GS.

*   **Large-scale NeRFs:**
    *   `Block-NeRF` and `Mega-NeRF` pioneered the "divide-and-conquer" approach for NeRFs. They partition a large scene (like a city block) into multiple smaller regions, each represented by its own independent NeRF model. This makes training manageable but can lead to inconsistencies between blocks and retains NeRF's slow rendering speed.
    *   `Switch-NeRF` uses a Mixture of Experts model to learn a scene decomposition, aiming for better scalability.

*   **Large-scale 3D-GS:**
    *   `VastGaussian` and `CityGaussian` apply the same "divide-and-conquer" strategy to 3D-GS. They split the scene into blocks and train an independent set of Gaussians for each. While this benefits from the speed of 3D-GS, the paper argues it suffers from a lack of cross-block interaction, leading to visual artifacts like the lighting discrepancies shown in Figure 1.
    *   `DOGS` introduces a distributed training algorithm (`ADMM`) to enforce consensus among Gaussians across different blocks, aiming to improve consistency. However, `Momentum-GS` argues that `DOGS` focuses on the optimization algorithm rather than optimizing the underlying representation for large scenes.

## 3.3. Technological Evolution
The field has evolved from traditional photogrammetry (`Structure-from-Motion`, `Multi-View Stereo`) towards neural representations for higher fidelity.
1.  **NeRF (2020):** Introduced high-quality neural view synthesis but was slow and limited to small scenes.
2.  **Large-Scale NeRFs (2022):** Methods like `Block-NeRF` and `Mega-NeRF` adapted NeRF to large scenes using a divide-and-conquer strategy.
3.  **3D Gaussian Splatting (2023):** Revolutionized the field with real-time rendering speeds and quality rivaling NeRF.
4.  **Large-Scale 3D-GS (2024):** Methods like `CityGaussian` and `VastGaussian` applied the divide-and-conquer approach to 3D-GS, but faced consistency issues.
5.  **Hybrid 3D-GS (2024):** Methods like `Scaffold-GS` introduced hybrid representations to reduce the memory footprint of 3D-GS.
6.  **Momentum-GS (This Paper):** Sits at the intersection of large-scale and hybrid 3D-GS. It proposes a novel training methodology to solve the consistency and scalability problems inherent in the divide-and-conquer strategy, especially when using hybrid representations.

## 3.4. Differentiation Analysis
*   **vs. `CityGaussian`/`VastGaussian`:** The key difference is the **information sharing mechanism**. While `CityGaussian` trains each block in isolation, `Momentum-GS` uses a shared student decoder and a global teacher decoder to enforce scene-wide consistency. This directly addresses the problem of boundary artifacts.
*   **vs. Simple Parallel Training:** Standard parallel training with a shared decoder is limited by the GPU count (number of blocks ≤ number of GPUs). `Momentum-GS` **breaks this link**. By training a subset of blocks at a time and using the momentum teacher to maintain a consistent global state, it can scale to an arbitrary number of blocks on a fixed number of GPUs.
*   **vs. `DOGS`:** `DOGS` enforces consistency through a complex optimization scheme (ADMM) on explicit Gaussians. `Momentum-GS` uses a much simpler and more elegant mechanism (self-distillation) and applies it to a `hybrid representation`, which provides additional benefits in terms of storage and memory efficiency.

# 4. Methodology
## 4.1. Principles
The core principle of `Momentum-GS` is to enable scalable and consistent large-scale scene reconstruction by combining a hybrid Gaussian representation with a novel teacher-student self-distillation training strategy. The method allows a large scene to be divided into many blocks ($n$) and trained on a smaller number of GPUs ($k$), overcoming the key limitations of previous approaches. Consistency is maintained by a slowly evolving "teacher" model that captures global scene information and guides the training of a "student" model shared across all blocks.

The architecture of the proposed method is illustrated in the figure below.

![该图像是示意图，展示了Momentum-GS方法在大规模场景重建中的工作流程。左侧展示了将稀疏体素划分为8个块的过程，右侧则展示了共享在线高斯解码器与动量高斯解码器的交互，以及在不同GPU上进行的训练过程。中间部分强调了重建一致性和动态调整块权重的机制，以提高重建精度。整体结构展示了方法如何解决块间一致性与资源利用的问题。](images/3.jpg)
*该图像是示意图，展示了Momentum-GS方法在大规模场景重建中的工作流程。左侧展示了将稀疏体素划分为8个块的过程，右侧则展示了共享在线高斯解码器与动量高斯解码器的交互，以及在不同GPU上进行的训练过程。中间部分强调了重建一致性和动态调整块权重的机制，以提高重建精度。整体结构展示了方法如何解决块间一致性与资源利用的问题。*

## 4.2. Core Methodology In-depth
The methodology of `Momentum-GS` can be broken down into two main components: Scene-Aware Momentum Self-Distillation and Reconstruction-guided Block Weighting.

### 4.2.1. Preliminaries: 3D Gaussian Splatting
The method builds upon the 3D-GS representation. Each point in the scene is modeled by a 3D Gaussian `G(x)` with center $\mu$ and covariance matrix $\Sigma$:
\$
G ( x ) = e ^ { - { \frac { 1 } { 2 } } ( x - \mu ) ^ { \top } \Sigma ^ { - 1 } ( x - \mu ) }
\$
For rendering, these 3D Gaussians are projected onto the 2D image plane and blended together using alpha blending to form the final pixel color $C(\mathbf{x}')$:
\$
C ( \mathbf { x } ^ { \prime } ) = \sum _ { i \in N } c _ { i } \sigma _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } )
\$
*   $C(\mathbf{x}')$: The color of a pixel $\mathbf{x}'$.
*   $N$: The set of all Gaussians that overlap with the pixel, sorted by depth.
*   $c_i$: The color of the $i$-th Gaussian (from view-dependent spherical harmonics).
*   $\sigma_i$: The contribution of the $i$-th Gaussian, which is a product of its learned opacity $\alpha_i$ and its 2D Gaussian function value at the pixel.

### 4.2.2. Scene-Aware Momentum Self-Distillation
This is the central component of the method, designed to ensure consistency while enabling scalability. It involves a student decoder ($D_s$), a teacher decoder ($D_t$), and a specific training procedure.

**1. Hybrid Representation with a Shared Student Decoder:**
Instead of storing explicit parameters for every Gaussian, `Momentum-GS` uses a hybrid representation. The scene is represented by a set of sparse `anchor` points, each with a feature vector $F$. A single, shared **student Gaussian decoder** ($D_s$), implemented as a small MLP, is used for all blocks. This decoder takes an anchor's feature, the viewing distance, and viewing direction as input and predicts the full Gaussian attributes (color, opacity, rotation, scale) on the fly.

**2. Decoupled Training with Sequential Parallelism:**
To train a scene divided into $n$ blocks on $k$ GPUs (where $n > k$), the method periodically samples a batch of $k$ blocks and distributes them across the $k$ GPUs for training. This process is repeated, cycling through all $n$ blocks over time. This **decouples the total number of blocks from the physical GPU count**.

**3. Student Training and Reconstruction Loss:**
For each active block, the shared student decoder $D_s$ predicts the Gaussians. These are rendered to produce an image, which is compared to the ground-truth image. The reconstruction loss $\mathcal{L}_{\mathrm{recons}}$ is calculated as a combination of an $\mathcal{L}_1$ loss and a structural similarity (SSIM) loss:
\$
\mathcal { L } _ { \mathrm { recon } } = \mathcal { L } _ { 1 } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } }
\$
*   $\mathcal{L}_1$: The sum of absolute differences between rendered and ground-truth pixel colors.
*   $\mathcal{L}_{\mathrm{SSIM}}$: The SSIM loss, which encourages structural similarity.
*   $\lambda_{\mathrm{SSIM}}$: A weighting factor.

    The gradients from all $k$ blocks are accumulated to update the parameters $\theta_s$ of the shared student decoder.

**4. The Teacher Decoder and Momentum Update:**
To prevent the student decoder from forgetting information about blocks that are not currently being trained, a **teacher Gaussian decoder** $D_t$ with parameters $\theta_t$ is maintained. The teacher acts as a stable, global model of the entire scene. It is **not** trained using backpropagation. Instead, its parameters $\theta_t$ are updated as an exponential moving average of the student's parameters $\theta_s$:
\$
\theta _ { t } \leftarrow m \cdot \theta _ { t } + ( 1 - m ) \cdot \theta _ { s }
\$
*   $m$: The momentum coefficient, set to a high value like 0.9. This ensures the teacher evolves smoothly and provides a consistent target for the student, aggregating knowledge from all blocks over time.

**5. Consistency Loss:**
To ensure the student learns this global knowledge, a consistency loss is introduced. This loss encourages the output of the student decoder $D_s$ to match the output of the teacher decoder $D_t$ for the same inputs.
\$
\mathcal { L } _ { \mathrm { consistency } } = \| D _ { t } ( f _ { b } , v _ { b } ; \theta _ { t } ) - D _ { s } ( f _ { b } , v _ { b } ; \theta _ { s } ) \| _ { 2 }
\$
*   $f_b$: The anchor feature for a sample in block $b$.
*   $v_b$: The viewing information for that sample.
*   $\theta_t, \theta_s$: Parameters of the teacher and student decoders, respectively.
    *(Note: The paper uses $D_m$ and $D_o$ in the formula, which appear to be typos for the teacher ($D_t$) and student ($D_s$, sometimes called online) decoders.)*

**6. Total Loss:**
The final loss function is a weighted sum of the reconstruction and consistency losses:
\$
\mathcal { L } = \mathcal { L } _ { \mathrm{recons} } + \lambda _ { \mathrm { consistency } } \mathcal { L } _ { \mathrm { consistency } } = \mathcal { L } _ { 1 } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } + \lambda _ { \mathrm { consistency } } \mathcal { L } _ { \mathrm { consistency } }
\$
This combined objective ensures the model both accurately reconstructs local details and maintains global consistency across the entire scene.

### 4.2.3. Reconstruction-guided Block Weighting
This mechanism dynamically adjusts the training focus to improve overall scene quality.

**1. Performance Tracking:** The model maintains a momentum-smoothed record of the `PSNR` and `SSIM` for each block to get a stable measure of its reconstruction quality.

**2. Identifying the Best-Performing Block:** At each stage, the block with the highest quality is identified, yielding reference values $\mathrm{PSNR}_{\max}$ and $\mathrm{SSIM}_{\max}$.

**3. Calculating Deviations:** For every other block $i$, its performance deviation from the best block is calculated:
*   PSNR deviation: $\delta_p = \mathrm{PSNR}_{\max} - \mathrm{PSNR}_i$
*   SSIM deviation: $\delta_s = \mathrm{SSIM}_{\max} - \mathrm{SSIM}_i$

    **4. Assigning Weights:** A weight $w_i$ is assigned to the loss of each block $i$. This weight is designed to be larger for blocks with higher deviations (i.e., worse performance). The formula is:
\$
w _ { i } = 2 - \exp \left( \frac { \delta _ { p } ^ { 2 } + \lambda \cdot \delta _ { s } ^ { 2 } } { - 2 \sigma ^ { 2 } } \right)
\$
*   The exponential term behaves like a Gaussian function. If a block's deviation is zero (it is the best block), the exponent is 0, $\exp(0)=1$, and its weight $w_i = 2-1 = 1$.
*   If a block's deviation is large, the negative exponent becomes large, the exponential term approaches 0, and its weight $w_i$ approaches 2.
    This weighting scheme effectively directs the shared decoder's capacity towards improving the underperforming parts of the scene, leading to better global consistency.

# 5. Experimental Setup
## 5.1. Datasets
The authors evaluated their method on six large-scale scenes from three challenging datasets, which primarily consist of aerial drone footage.
*   **Mill19 Dataset:** Contains scenes like `Building` and `Rubble`.
*   **UrbanScene3D Dataset:** Contains scenes like `Campus`, `Residence`, and `Sci-Art`.
*   **MatrixCity Dataset:** A particularly massive dataset, with the `Small City` scene covering 2.7 square kilometers.

    For most scenes, images were downsampled by a factor of 4. For the huge `MatrixCity` dataset, images were resized to a width of 1,600 pixels. This setup is consistent with prior work and provides a rigorous testbed for large-scale reconstruction methods.

## 5.2. Evaluation Metrics
The performance of the reconstructions was measured using three standard metrics:

1.  **PSNR (Peak Signal-to-Noise Ratio):**
    *   **Conceptual Definition:** PSNR measures the pixel-wise accuracy of a reconstructed image compared to a ground-truth image. It is based on the Mean Squared Error (MSE) between the images. A higher PSNR value indicates a better reconstruction with less error. It is measured in decibels (dB).
    *   **Mathematical Formula:**
        \$
        \text{PSNR} = 20 \cdot \log_{10}(\text{MAX}_I) - 10 \cdot \log_{10}(\text{MSE})
        \$
    *   **Symbol Explanation:**
        *   $\text{MAX}_I$: The maximum possible pixel value of the image (e.g., 255 for an 8-bit image).
        *   $\text{MSE}$: The Mean Squared Error between the ground-truth image $I$ and the reconstructed image $K$, calculated as `\frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2`.

2.  **SSIM (Structural Similarity Index Measure):**
    *   **Conceptual Definition:** SSIM is a perceptual metric that measures the similarity between two images based on human perception. Unlike PSNR, which treats all pixel errors equally, SSIM evaluates similarity in terms of three components: luminance, contrast, and structure. Its value ranges from -1 to 1, where 1 indicates a perfect match.
    *   **Mathematical Formula:**
        \$
        \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
        \$
    *   **Symbol Explanation:**
        *   $\mu_x, \mu_y$: The average of images $x$ and $y$.
        *   $\sigma_x^2, \sigma_y^2$: The variance of images $x$ and $y$.
        *   $\sigma_{xy}$: The covariance of $x$ and $y$.
        *   $c_1, c_2$: Small constants to stabilize the division.

3.  **LPIPS (Learned Perceptual Image Patch Similarity):**
    *   **Conceptual Definition:** LPIPS is a more advanced perceptual metric that aims to better align with human judgment of image similarity. It uses a pre-trained deep convolutional neural network (e.g., VGG or AlexNet). To compare two images, they are passed through the network, and the LPIPS distance is calculated as the weighted sum of distances between their feature activations at different layers. A **lower** LPIPS score indicates that the two images are more perceptually similar.
    *   **Mathematical Formula:** There isn't a single closed-form equation. The distance is computed as:
        \$
        d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} || w_l \odot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l) ||_2^2
        \$
    *   **Symbol Explanation:**
        *   $d(x, x_0)$: The LPIPS distance between images $x$ and $x_0$.
        *   $l$: Index of a layer in the deep network.
        *   $\hat{y}^l, \hat{y}_0^l$: Feature activations for the images at layer $l$.
        *   $w_l$: A learned weight to scale the contribution of each layer.

## 5.3. Baselines
The paper compares `Momentum-GS` against a comprehensive set of representative baseline models, including:
*   **NeRF-based methods:** `Mega-NeRF`, `Switch-NeRF`.
*   **Vanilla 3D-GS:** The original `3D-GS` method.
*   **Large-scale 3D-GS methods:** `VastGaussian`, `CityGaussian`, and `DOGS`.

    This selection allows for a thorough comparison against both the foundational technology and direct competitors in the large-scale reconstruction domain.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The experimental results strongly validate the effectiveness of `Momentum-GS`.

**Quantitative Results:**
The following are the results from Table 1 and Table 2 of the original paper, comparing `Momentum-GS` to baselines on various large-scale scenes.

<table>
<thead>
<tr>
<th rowspan="2">Scene</th>
<th colspan="3">Building</th>
<th colspan="3">Rubble</th>
<th colspan="3">Campus</th>
<th colspan="3">Residence</th>
<th colspan="3">Sci-Art</th>
</tr>
<tr>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
<th>PSNR ↑</th>
<th>SSIM ↑</th>
<th>LPIPS ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Mega-NeRF [55]</td>
<td>20.93</td>
<td>0.547</td>
<td>0.504</td>
<td>24.06</td>
<td>0.553</td>
<td>0.516</td>
<td>23.42</td>
<td>0.537</td>
<td>0.636</td>
<td>22.08</td>
<td>0.628</td>
<td>0.489</td>
<td><b>25.60</b></td>
<td>0.770</td>
<td>0.390</td>
</tr>
<tr>
<td>Switch-NRF [37]</td>
<td>21.54</td>
<td>0.579</td>
<td>0.474</td>
<td>24.31</td>
<td>0.562</td>
<td>0.496</td>
<td>23.62</td>
<td>0.541</td>
<td>0.616</td>
<td>22.57</td>
<td>0.654</td>
<td>0.457</td>
<td><b>26.52</b></td>
<td>0.795</td>
<td>0.360</td>
</tr>
<tr>
<td>3D-GS [20]</td>
<td>22.53</td>
<td>0.738</td>
<td>0.214</td>
<td>25.51</td>
<td>0.725</td>
<td>0.316</td>
<td>23.67</td>
<td>0.688</td>
<td>0.347</td>
<td>22.36</td>
<td>0.745</td>
<td>0.247</td>
<td>24.13</td>
<td>0.791</td>
<td>0.262</td>
</tr>
<tr>
<td>VastGaussian [29]</td>
<td>21.80</td>
<td>0.728</td>
<td>0.225</td>
<td>25.20</td>
<td>0.742</td>
<td>0.264</td>
<td>23.82</td>
<td>0.695</td>
<td>0.329</td>
<td>21.01</td>
<td>0.699</td>
<td>0.261</td>
<td>22.64</td>
<td>0.761</td>
<td>0.261</td>
</tr>
<tr>
<td>CityGaussian [33]</td>
<td>22.70</td>
<td>0.774</td>
<td>0.246</td>
<td>26.45</td>
<td>0.809</td>
<td>0.232</td>
<td>22.80</td>
<td>0.662</td>
<td>0.437</td>
<td>23.35</td>
<td>0.822</td>
<td>0.211</td>
<td>24.49</td>
<td>0.843</td>
<td>0.232</td>
</tr>
<tr>
<td>DOGS [9]</td>
<td>22.73</td>
<td>0.759</td>
<td>0.204</td>
<td>25.78</td>
<td>0.765</td>
<td>0.257</td>
<td>24.01</td>
<td>0.681</td>
<td>0.377</td>
<td>21.94</td>
<td>0.740</td>
<td>0.244</td>
<td>24.42</td>
<td>0.804</td>
<td>0.219</td>
</tr>
<tr>
<td>Momentum-GS (Ours)</td>
<td><b>23.65</b></td>
<td><b>0.813</b></td>
<td><b>0.194</b></td>
<td><b>26.66</b></td>
<td><b>0.826</b></td>
<td><b>0.200</b></td>
<td><b>24.34</b></td>
<td><b>0.760</b></td>
<td><b>0.290</b></td>
<td><b>23.37</b></td>
<td><b>0.828</b></td>
<td><b>0.196</b></td>
<td>25.06</td>
<td><b>0.860</b></td>
<td><b>0.204</b></td>
</tr>
</tbody>
</table>

The following are the results from Table 2 of the original paper:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :--- | :--- |
| 3D-GS [20] | 27.36 | 0.818 | 0.237 |
| VastGaussian [29] | 28.33 | 0.835 | 0.220 |
| CityGaussian [33] | 28.61 | 0.868 | 0.205 |
| DOGS [9] | 28.58 | 0.847 | 0.219 |
| Momentum-GS (Ours) | **29.11** | **0.881** | **0.180** |

*   **Overall Superiority:** `Momentum-GS` consistently achieves the best or second-best scores across all scenes and metrics. The improvements are particularly strong in **LPIPS**, which suggests that the reconstructions are perceptually more realistic to humans. For example, on the `MatrixCity` dataset, it improves LPIPS from 0.205 (`CityGaussian`) to 0.180, a significant jump.
*   **Sci-Art Anomaly:** On the `Sci-Art` scene, NeRF-based methods achieve higher PSNR. The authors explain this is likely because the source images are blurry. NeRFs tend to produce smoother (blurrier) outputs, which happen to have a lower pixel-wise error (MSE) against the blurry ground truth, artificially inflating the PSNR score. However, on the perceptual metrics SSIM and LPIPS, `Momentum-GS` is clearly superior, indicating it produces sharper and more structurally correct images.

**Visualization Results:**
The visual comparisons in Figure 4 and Figure 5 of the paper reinforce the quantitative findings. `Momentum-GS` produces noticeably sharper images with better-preserved fine details (e.g., building facades, foliage) compared to other methods, which often exhibit blurriness or lose structural integrity.

![该图像是比较不同重建技术的示意图，包括Mega-NeRF、3DGS、CityGaussian、DOGS、Momentum-GS（我们的算法）和真实图像（Ground Truth）。图中展示了建筑、碎石、校园、住宅和科学艺术五种场景，左侧小框显示了各方法在重建质量上的差异，Momentum-GS相比其他方法显示了更好的重建精度。](images/4.jpg)

**Performance and Efficiency:**
The following are the results from Table 3 of the original paper:

| Method | FPS ↑ | Mem ↓ |
| :--- | :--- | :--- |
| 3D-GS | 45.57 | 6.31 |
| VastGaussian | 40.04 | 6.99 |
| CityGaussian | 26.10 | 14.68 |
| DOGS | 48.34 | 5.82 |
| Momentum-GS (Ours) | **59.91** | **4.62** |

On the extremely large `MatrixCity` scene, `Momentum-GS` not only produces higher quality results but is also more efficient. It achieves the **highest rendering framerate (FPS)** and consumes the **least GPU memory (Mem)** during evaluation. This demonstrates the power of the hybrid representation, which generates Gaussians on-the-fly, avoiding the need to load a massive number of explicit Gaussians into memory.

## 6.2. Ablation Studies / Parameter Analysis
The authors conduct a thorough set of ablation studies to validate each component of their proposed method.

**Parallel vs. Independent Training (Table 5):**
The following are the results from Table 5 of the original paper:

| Training strategy | #Block | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :--- | :--- | :--- |
| (a) baseline | 1 | 22.25 | 0.742 | 0.272 |
| (b) w/ Parallel training | 4 | 23.10 | 0.790 | 0.221 |
| (c) w/ Independent training | 4 | 22.85 | 0.781 | 0.229 |
| (d) w/ Independent training | 8 | 23.23 | 0.796 | 0.211 |
| (e) w/ momentum self-distill. | 8 | 23.56 | 0.806 | 0.205 |
| (f) Full | 8 | **23.65** | **0.813** | **0.194** |

This study is crucial. It shows that:
*   Parallel training with a shared decoder (b) is better than independent training (c) for the same number of blocks, because the decoder sees more diverse data.
*   However, independent training can use more blocks (d), which can surpass a resource-limited parallel setup (b). This highlights the core problem.
*   `Momentum-GS` with self-distillation (e) significantly outperforms independent training (d) with the same number of blocks, proving the benefit of the global teacher model.
*   Adding the reconstruction-guided block weighting (f) provides a final performance boost.

**Effectiveness of Self-Distillation (Table 7):**
This study in the supplementary material further isolates the benefit of the momentum self-distillation. It shows that when training 8 blocks on only 4 GPUs (alternating), simply adding the self-distillation mechanism (d) boosts performance significantly, approaching the quality of training on 8 GPUs (b). This confirms that the method successfully mitigates the hardware constraint.

**Block Weighting Strategy (Table 4):**
This ablation shows that using a combination of `PSNR` and `SSIM` to guide the block weighting yields better results than using either metric alone, validating the design choice.

**Scalability with Number of Blocks (Table 6):**
The following are the results from Table 6 of the original paper:

| Method | #Block | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :--- | :--- | :--- |
| CityGaussian | 32 | 28.61 | 0.868 | 0.205 |
| Momentum-GS (Ours) | 4 | 28.93 | 0.870 | 0.203 |
| Momentum-GS (Ours) | 8 | 29.11 | 0.881 | 0.180 |
| Momentum-GS (Ours) | 16 | **29.15** | **0.884** | **0.172** |

This experiment demonstrates the excellent scalability of `Momentum-GS`. While keeping the GPU count fixed, increasing the number of blocks from 4 to 16 consistently improves reconstruction quality. Notably, `Momentum-GS` with only 8 blocks already outperforms `CityGaussian` which uses 32 blocks.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
The paper introduces `Momentum-GS`, a novel framework for high-quality, large-scale 3D scene reconstruction. By leveraging a **hybrid representation** combined with **momentum-based self-distillation**, the method successfully addresses the critical challenges of consistency and scalability in block-wise training. The introduction of a momentum-updated **teacher decoder** provides a stable global reference that guides the training of all blocks, ensuring spatial coherence. This is further enhanced by a **reconstruction-guided block weighting** mechanism that dynamically focuses training on weaker areas of the scene. The experimental results demonstrate that `Momentum-GS` establishes a new state of the art, achieving superior visual quality and efficiency compared to previous methods.

## 7.2. Limitations & Future Work
The paper does not explicitly state its limitations, but some can be inferred:
*   **Dependence on Initial SfM:** Like most methods in this domain, `Momentum-GS` relies on an initial point cloud generated by Structure-from-Motion (SfM) from COLMAP. The final reconstruction quality is therefore inherently limited by the quality of this initial camera pose estimation and sparse reconstruction.
*   **Hyperparameter Sensitivity:** The method introduces several new hyperparameters, such as the momentum coefficient $m$, the consistency loss weight $\lambda_{consistency}$, and parameters for the block weighting formula ($\lambda, \sigma$). While the ablation studies show robustness, optimal performance on new and diverse scenes might require careful tuning.
*   **Training Time:** The paper focuses on inference speed and memory but does not provide a comparison of total training time. The sequential training of blocks, while scalable, might lead to longer overall training durations compared to fully parallel methods (if sufficient hardware were available).
*   **Block Partitioning Strategy:** The current method uses a simple grid-based partitioning. A more semantic or content-aware partitioning strategy could potentially lead to better results by creating more coherent blocks.

    Future work could explore learnable block weighting schemes, end-to-end optimization of camera poses alongside the scene representation, or applying the momentum-distillation concept to other large-scale decomposition problems in graphics and vision.

## 7.3. Personal Insights & Critique
`Momentum-GS` presents a very elegant and effective solution to a well-known problem.
*   **Key Insight:** The application of momentum self-distillation, a technique popularized in self-supervised representation learning (e.g., MoCo), to the domain of 3D reconstruction is highly innovative. It provides a simple yet powerful way to enforce global consistency in a decoupled training environment, which is far more straightforward than complex optimization schemes like ADMM.
*   **Practical Significance:** By decoupling the number of scene blocks from the GPU count, the method makes high-quality reconstruction of massive scenes practical for users with limited hardware resources. The demonstrated improvements in efficiency (FPS, memory, storage) further underscore its practical value.
*   **Critique and Nuances:**
    *   While the paper claims efficiency gains, the VRAM usage in Table 10 of the supplement shows a mixed picture. For `Residence` and `Sci-Art`, `Momentum-GS` uses comparable or even more memory than `CityGaussian` during inference. This suggests that the overhead of the neural decoder can sometimes be significant, and the efficiency benefits are most pronounced on extremely large scenes like `MatrixCity`.
    *   The block weighting formula, while effective, feels somewhat heuristic. The choice of the "2 - exp(...)" form and its parameters could be explored further. A learnable weighting mechanism might offer a more principled approach.
    *   The paper's core strength is its novel training methodology. This idea of using a momentum teacher for consistency in a "divide-and-conquer" setting is highly transferable and could inspire solutions in other areas, such as large-scale video processing, federated learning, or panoramic image stitching, where sub-problems must be solved independently but combine into a coherent whole.