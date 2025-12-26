# 1. Bibliographic Information

## 1.1. Title
Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs

## 1.2. Authors
Brandon Smart (Active Vision Lab, University of Oxford), Chuanxia Zheng (Visual Geometry Group, University of Oxford), Iro Laina (Visual Geometry Group, University of Oxford), Victor Adrian Prisacariu (Active Vision Lab, University of Oxford).

## 1.3. Journal/Conference
The paper was published on arXiv (a prominent preprint server for rapid dissemination of research in Computer Science and Computer Vision) on August 25, 2024. Given the affiliations (University of Oxford's Active Vision Lab and VGG), this work carries significant weight in the research community.

## 1.4. Publication Year
2024

## 1.5. Abstract
Splatt3R is a pose-free, feed-forward method designed for 3D reconstruction and novel view synthesis (NVS) from stereo image pairs. Unlike traditional methods that require camera parameters (calibration) or iterative optimization, Splatt3R predicts 3D Gaussian Splats directly from uncalibrated natural images. It builds upon the MASt3R geometry reconstruction model, extending it to handle both structure and appearance. The method uses a two-stage training approach: first optimizing for 3D geometry and then for novel view synthesis. A novel loss masking strategy is introduced to handle extrapolated viewpoints effectively. Splatt3R achieves real-time rendering and can reconstruct scenes at 4 frames per second (FPS) at $512 \times 512$ resolution.

## 1.6. Original Source Link
*   **PDF Link:** [https://arxiv.org/pdf/2408.13912v2.pdf](https://arxiv.org/pdf/2408.13912v2.pdf)
*   **Publication Status:** Preprint (v2 updated August 2024).

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem Splatt3R addresses is the difficulty of performing high-quality 3D reconstruction and **Novel View Synthesis (NVS)**—generating images of a scene from viewpoints not present in the original data—using only a few uncalibrated images.

Historically, this required:
1.  **Dense Image Sets:** Dozens or hundreds of images of the same scene.
2.  **Known Camera Poses:** Precise information about where each photo was taken (intrinsics and extrinsics), often obtained through slow **Structure-from-Motion (SfM)** algorithms.
3.  **Iterative Optimization:** Slow, per-scene training (like original NeRF or 3D Gaussian Splatting) that cannot generalize to new, unseen scenes.

    Splatt3R's innovation lies in its "zero-shot" and "pose-free" nature. It aims to take just two random photos and immediately produce a 3D model that can be viewed from any angle without needing to know the camera settings or wait for a long training process.

## 2.2. Main Contributions / Findings
*   **Pose-free, Feed-forward Reconstruction:** It is the first method to generate 3D Gaussian Splats from uncalibrated stereo pairs in a single forward pass of a neural network.
*   **Foundation Model Extension:** It leverages **MASt3R**, a state-of-the-art geometry model, and expands its architecture to predict appearance attributes (color, opacity, etc.) alongside geometry.
*   **Novel Loss Masking Strategy:** It introduces a method to mask out parts of the scene during training that were never visible in the input images. This prevents the model from "guessing" unseen areas, which improves performance on wide-angle, extrapolated views.
*   **Efficiency:** The model operates at 4 FPS for reconstruction and allows for real-time rendering of the resulting 3D Gaussians.
*   **Robust Generalization:** It demonstrates an ability to reconstruct "in-the-wild" scenes (like outdoor environments or mobile phone photos) even when the two input images have very little overlap.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
*   **3D Gaussian Splatting (3D-GS):** A method where a 3D scene is represented by millions of tiny, "fuzzy" 3D ellipses (Gaussians). Each Gaussian has a position, rotation, scale, opacity, and color. They are "splatted" (projected) onto a 2D screen to render an image very quickly.
*   **Uncalibrated Images:** Photos taken without knowing the camera's internal properties (focal length, sensor center) or its external position/orientation in space.
*   **Feed-forward Model:** A neural network that produces a result in one go (like an image filter), rather than through an iterative process of trial and error (optimization).
*   **Stereo Pairs:** Two images of the same scene taken from slightly different positions, allowing for the calculation of depth (like human binocular vision).

## 3.2. Previous Works
*   **NeRF (Neural Radiance Fields):** A precursor to 3D-GS that represents scenes as continuous volumes. It is high-quality but very slow to train and render.
*   **DUSt3R & MASt3R:** These are "foundation models" for 3D vision. Instead of traditional matching, they use Transformers to predict "point maps" (a 3D coordinate for every pixel) directly from image pairs. Splatt3R uses MASt3R as its structural backbone.
*   **pixelSplat & MVSplat:** Recent "generalizable" Gaussian methods. While fast, they **require** known camera poses to function, making them unusable for random "in-the-wild" photos.

## 3.3. Technological Evolution
The field has moved from **per-scene optimization with many images** (NeRF/3D-GS) to **generalizable models with sparse calibrated views** (pixelSplat), and now with Splatt3R, to **generalizable models with sparse uncalibrated views**. This represents the "holy grail" of 3D vision: reconstruction from any two casual photos.

## 3.4. Differentiation Analysis
The primary difference is **pose independence**. Most 3D-GS models use camera rays (calculated from poses) to place their Gaussians. Splatt3R uses the learned geometric priors of MASt3R to predict 3D positions directly in a global coordinate frame, bypassing the need for an explicit camera pose estimation step.

# 4. Methodology

## 4.1. Principles
The core idea of Splatt3R is to treat 3D reconstruction as a **dense prediction task**. Instead of matching features and triangulating, the model uses a large Transformer to "imagine" the 3D structure and appearance of every pixel simultaneously. By building on MASt3R, it inherits a strong understanding of 3D shapes.

## 4.2. Core Methodology In-depth (Layer by Layer)

The following figure (Figure 2 from the original paper) shows the system architecture:

![该图像是示意图，展示了 Splatt3R 方法的模块结构，包括两个输入图像的 ViT 编码器和不同的特征匹配头。图中体现了点云生成和 3D Gaussian Splat 相关的损失计算过程。关键公式涉及 Gaussian 参数的描述。](images/2.jpg)
*该图像是示意图，展示了 Splatt3R 方法的模块结构，包括两个输入图像的 ViT 编码器和不同的特征匹配头。图中体现了点云生成和 3D Gaussian Splat 相关的损失计算过程。关键公式涉及 Gaussian 参数的描述。*

### 4.2.1. Feature Extraction and Interaction
The process begins with two input images $\mathbf{I}^1, \mathbf{I}^2$. These are fed into a **Vision Transformer (ViT)** encoder. A Transformer decoder then performs cross-attention, allowing the features of image 1 to "talk" to image 2. This helps the network understand the relationship and depth between the two views.

### 4.2.2. The Prediction Heads
The decoder outputs features that are sent to three parallel "heads":
1.  **Point & Confidence Head:** Inherited from MASt3R. For each pixel $i$ in view $v \in \{1, 2\}$, it predicts a 3D location $\hat{X}_i^v$ and a confidence score $C_i^v$.
2.  **Feature Matching Head:** (Optional, used for alignment).
3.  **Gaussian Head (The Splatt3R Addition):** This head uses a **Dense Prediction Transformer (DPT)** architecture to output the remaining attributes for each 3D Gaussian:
    *   **Covariance:** Parameterized as a rotation quaternion $q \in \mathbb{R}^4$ and scale $s \in \mathbb{R}^3$.
    *   **Appearance:** Spherical Harmonics $S \in \mathbb{R}^{3 \times d}$ (for color) and opacity $\alpha \in \mathbb{R}$.
    *   **Offset:** $\Delta \in \mathbb{R}^3$.

        The final mean position of the Gaussian for a pixel is calculated as:
\$
\mu = X + \Delta
\$
where $X$ is the base point predicted by the geometry head. This allows the model to refine the 3D position specifically for rendering purposes.

### 4.2.3. Training Stage 1: Geometry Loss
To avoid the local minima (errors where the model gets stuck) common in 3D-GS, the model is first trained to get the 3D points right. The geometry training objective $L_{pts}$ is defined as:
\$
L_{pts} = \sum_{v \in \{1, 2\}} \sum_{i} C_i^v L_{regr}(v, i) - \gamma \log(C_i^v)
\$
where the regression loss $L_{regr}$ is:
\$
L_{regr}(v, i) = \left| \frac{1}{z} X_i^v - \frac{1}{\bar{z}} \hat{X}_i^v \right|
\$
*   $X_i^v$ is the ground truth point.
*   $\hat{X}_i^v$ is the predicted point.
*   $z$ and $\bar{z}$ are normalization factors.
*   $C_i^v$ is the confidence; the log term $-\gamma \log(C_i^v)$ prevents the model from simply setting all confidence scores to zero to "cheat" the loss.

### 4.2.4. Training Stage 2: Novel View Synthesis & Loss Masking
Once geometry is stable, the model is trained to render new views. However, if a target view sees a part of the scene that wasn't in the input images, the loss would be noisy. Splatt3R introduces a **Loss Mask** $M$.

The following figure (Figure 3 from the original paper) illustrates the masking approach:

![Figure 3. Our loss masking approach. Valid pixels are considered to be those that are: inside the frustum of at least one of the views, have their reprojected depth match the ground truth, and are considered valid pixels with valid depth in their dataset.](images/3.jpg)
*该图像是示意图，展示了通过目标视图和上下文视图计算有效像素的过程。这些有效像素需满足在视锥内、具有匹配深度，并被视为有效深度点。最后，生成的损失掩膜将用于后续的3D重建任务。*

The mask $M$ identifies pixels that are visible in at least one input view by reprojecting points and checking depth consistency. The final masked reconstruction loss $\mathcal{L}$ is:
\$
\mathcal{L} = \lambda_{MSE} L_{MSE} (M \odot \hat{\mathbf{I}}, M \odot \mathbf{I}) + \lambda_{LPIPS} L_{LPIPS} (M \odot \hat{\mathbf{I}}, M \odot \mathbf{I})
\$
*   $\hat{\mathbf{I}}$ is the rendered image, and $\mathbf{I}$ is the ground truth target image.
*   $\odot$ is the element-wise multiplication (applying the mask).
*   $L_{MSE}$ is the Mean Squared Error.
*   $L_{LPIPS}$ is a perceptual loss that checks if the images "look" similar to a human, even if pixel values differ slightly.

# 5. Experimental Setup

## 5.1. Datasets
*   **ScanNet++:** A high-fidelity dataset of over 450 indoor scenes. It provides ground truth 3D geometry from laser scans and high-resolution images.
*   **In-the-wild data:** The authors also tested on casual photos taken with a mobile phone to prove generalization.

## 5.2. Evaluation Metrics
1.  **PSNR (Peak Signal-to-Noise Ratio):**
    *   **Concept:** Measures the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher is better (less noise).
    *   **Formula:** $\mathrm{PSNR} = 10 \cdot \log_{10}\left(\frac{\mathit{MAX}_I^2}{\mathit{MSE}}\right)$
    *   **Symbols:** $\mathit{MAX}_I$ is the maximum pixel value (usually 255 or 1.0); $\mathit{MSE}$ is the mean squared error.
2.  **SSIM (Structural Similarity Index Measure):**
    *   **Concept:** Quantifies the perceived quality of digital images by comparing luminance, contrast, and structure. Scores range from 0 to 1 (1 is perfect).
    *   **Formula:** $\mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$
    *   **Symbols:** $\mu$ is the mean; $\sigma^2$ is the variance; $\sigma_{xy}$ is the covariance; $c$ are constants.
3.  **LPIPS (Learned Perceptual Image Patch Similarity):**
    *   **Concept:** Uses a deep neural network to measure how similar two images look to humans. Lower is better.

## 5.3. Baselines
*   **MASt3R (Point Cloud):** Rendering the raw 3D points as colored dots.
*   **pixelSplat:** A state-of-the-art feed-forward Gaussian model. It was tested both with Ground Truth (GT) poses and with poses estimated by MASt3R to see if it could handle uncalibrated inputs.

# 6. Results & Analysis

## 6.1. Core Results Analysis
Splatt3R consistently outperforms the baselines. Notably, it performs better than `pixelSplat` even when `pixelSplat` is given the perfect ground truth camera poses. This suggests that the Splatt3R architecture is inherently better at handling wide-baseline stereo pairs.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Close (φ = 0.9, ψ = 0.9)</th>
<th colspan="3">Medium (φ = 0.7, ψ = 0.7)</th>
<th colspan="3">Wide (φ = 0.5, ψ = 0.5)</th>
<th colspan="3">Very Wide (φ = 0.3, ψ = 0.3)</th>
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
</tr>
</thead>
<tbody>
<tr>
<td>Splatt3R (Ours)</td>
<td>19.66 (14.72)</td>
<td>0.757</td>
<td>0.234 (0.237)</td>
<td>19.66 (14.38)</td>
<td>0.770</td>
<td>0.229 (0.243)</td>
<td>19.41 (13.72)</td>
<td>0.783</td>
<td>0.220 (0.247)</td>
<td>19.18 (12.94)</td>
<td>0.794</td>
<td>0.209 (0.258)</td>
</tr>
<tr>
<td>MASt3R (Point Cloud)</td>
<td>18.56 (13.57)</td>
<td>0.708</td>
<td>0.278 (0.283)</td>
<td>18.51 (12.96)</td>
<td>0.718</td>
<td>0.259 (0.280)</td>
<td>18.73 (12.50)</td>
<td>0.739</td>
<td>0.245 (0.293)</td>
<td>18.44 (11.27)</td>
<td>0.758</td>
<td>0.242 (0.322)</td>
</tr>
<tr>
<td>pixelSplat (MASt3R cams)</td>
<td>15.48 (10.53)</td>
<td>0.602</td>
<td>0.439 (0.447)</td>
<td>15.96 (10.64)</td>
<td>0.648</td>
<td>0.379 (0.405)</td>
<td>15.94 (10.14)</td>
<td>0.675</td>
<td>0.343 (0.394)</td>
<td>16.46 (10.12)</td>
<td>0.708</td>
<td>0.302 (0.373)</td>
</tr>
<tr>
<td>pixelSplat (GT cams)</td>
<td>15.67 (10.71)</td>
<td>0.609</td>
<td>0.436 (0.443)</td>
<td>15.92 (10.61)</td>
<td>0.643</td>
<td>0.381 (0.407)</td>
<td>16.08 (10.33)</td>
<td>0.672</td>
<td>0.407 (0.392)</td>
<td>16.56 (10.20)</td>
<td>0.709</td>
<td>0.299 (0.370)</td>
</tr>
</tbody>
</table>

*(Note: Values in parentheses are metrics averaged only over the pixels within the loss mask).*

## 6.2. Ablation Studies
The authors tested what happens if they remove parts of the model.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Close (φ = 0.9, ψ = 0.9)</th>
<th colspan="3">Medium (φ = 0.7, ψ = 0.7)</th>
<th colspan="3">Wide (φ = 0.5, ψ = 0.5)</th>
<th colspan="3">Very Wide (φ = 0.3, ψ = 0.3)</th>
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
</tr>
</thead>
<tbody>
<tr>
<td>Ours</td>
<td>19.66 (14.72)</td>
<td>0.757</td>
<td>0.234 (0.237)</td>
<td>19.66 (14.38)</td>
<td>0.770</td>
<td>0.229 (0.243)</td>
<td>19.41 (13.72)</td>
<td>0.783</td>
<td>0.220 (0.247)</td>
<td>19.18 (12.94)</td>
<td>0.794</td>
<td>0.209 (0.258)</td>
</tr>
<tr>
<td>+ Finetune w/ MASt3R</td>
<td>20.97 (16.03)</td>
<td>0.780</td>
<td>0.199 (0.201)</td>
<td>20.41 (15.13)</td>
<td>0.781</td>
<td>0.214 (0.226)</td>
<td>20.00 (14.32)</td>
<td>0.793</td>
<td>0.207 (0.232)</td>
<td>19.69 (13.45)</td>
<td>0.803</td>
<td>0.197 (0.241)</td>
</tr>
<tr>
<td>- LPIPS Loss</td>
<td>19.62 (14.68)</td>
<td>0.763</td>
<td>0.277 (0.282)</td>
<td>19.65 (14.37)</td>
<td>0.776</td>
<td>0.261 (0.278)</td>
<td>19.41 (13.73)</td>
<td>0.787</td>
<td>0.245 (0.278)</td>
<td>19.22 (12.98)</td>
<td>0.797</td>
<td>0.230 (0.285)</td>
</tr>
<tr>
<td>- Loss Masking</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
</tbody>
</table>

**Key Finding:** Removing **Loss Masking** caused the model to fail (Gaussians grew infinitely in size), proving it is critical for training on uncalibrated, wide-baseline images.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
Splatt3R represents a major step forward in democratic 3D reconstruction. By decoupling the reconstruction process from known camera poses and using a powerful Transformer-based "foundation" model (MASt3R), it enables anyone to create a high-quality, renderable 3D scene from just two photos. The introduction of loss masking allows the model to handle wide baselines and extrapolated views that previous models could not touch.

## 7.2. Limitations & Future Work
*   **Unseen Regions:** The model does not attempt to "hallucinate" or guess parts of the scene that are completely hidden. While this leads to more accurate reconstructions of visible areas, it results in "holes" in the 3D model if the user wants to see the back of an object.
*   **View-Dependent Color:** The authors found that Spherical Harmonics (which allow color to change based on viewing angle) actually hurt performance, likely due to overfitting. Future work could improve how view-dependency is learned.

## 7.3. Personal Insights & Critique
The decision to build on MASt3R is brilliant. It recognizes that **geometry is the hardest part** of 3D reconstruction. By using a model that already "knows" 3D shapes, Splatt3R can focus on the rendering attributes. The loss masking strategy is a clever, simple solution to the problem of "garbage in, garbage out" when training on images with limited overlap. However, the reliance on a two-image input might be restrictive; extending this to "n-unposed-images" would be the next logical step. The model's speed (4 FPS) is impressive but still short of "instant" for mobile applications, suggesting room for further optimization.