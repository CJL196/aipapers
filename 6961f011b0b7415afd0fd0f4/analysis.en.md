# 1. Bibliographic Information

## 1.1. Title
RayZer: A Self-supervised Large View Synthesis Model

## 1.2. Authors
Hanwen Jiang (The University of Texas at Austin), Hao Tan (Adobe Research), Peng Wang (Adobe Research), Haian Jin (Cornell University), Yue Zhao (UT Austin), Sai Bi (Adobe Research), Kai Zhang (Adobe Research), Fujun Luan (Adobe Research), Kalyan Sunkavalli (Adobe Research), Qixing Huang (UT Austin), and Georgios Pavlakos (UT Austin).

## 1.3. Journal/Conference
This paper was published at **CVPR 2025** (Conference on Computer Vision and Pattern Recognition). CVPR is widely considered the top-tier conference in computer vision, known for its extreme rigor and significant impact on the field.

## 1.4. Publication Year
The paper was published on May 1, 2025 (UTC).

## 1.5. Abstract
The authors present `RayZer`, a self-supervised multi-view 3D vision model. Unlike traditional models that require 3D supervision (such as camera poses and scene geometry), `RayZer` is trained entirely without them. It takes unposed and uncalibrated images as input, recovers camera parameters, reconstructs a scene representation, and synthesizes novel views. Its "emerging 3D awareness" comes from two factors: a self-supervised framework that disentangles camera and scene representations, and a transformer-based architecture where the only 3D prior is the ray structure. `RayZer` demonstrates performance comparable to or better than "oracle" methods that use ground-truth pose annotations.

## 1.6. Original Source Link
- **PDF Link:** [https://arxiv.org/pdf/2505.00702v1.pdf](https://arxiv.org/pdf/2505.00702v1.pdf)
- **Publication Status:** Preprint/Conference Paper.

  ---

# 2. Executive Summary

## 2.1. Background & Motivation
In the field of 3D computer vision, the current paradigm relies heavily on **Supervised Learning**. To train a model to understand a 3D scene from 2D images, researchers usually need "Ground Truth" (GT) data—specifically, the exact position of the camera for every image (camera poses) and the underlying shape of the scene (geometry).

**The Problem:** Obtaining these labels is difficult. Researchers often use tools like `COLMAP` (a Structure-from-Motion software), which is slow, computationally expensive, and often produces "noisy" or slightly incorrect poses. This reliance on 3D labels limits the ability to train models on the vast amount of unlabeled video data available on the internet.

**The Innovation:** `RayZer` asks: "How far can we push a 3D model without any 3D supervision?" The paper proposes a **Self-supervised** approach. Instead of being told where the camera is, the model *predicts* the camera pose and then uses its own prediction to try and reconstruct the image. If the reconstruction is good, the pose must have been somewhat accurate.

## 2.2. Main Contributions / Findings
1.  **Zero 3D Supervision:** `RayZer` is a large-scale model trained without any camera pose or geometry annotations. It learns entirely from 2D images.
2.  **Disentanglement Framework:** The model successfully separates "what the scene looks like" (scene representation) from "where the camera is" (camera parameters).
3.  **Ray-Based Transformer:** The authors designed a transformer model that uses the physical concept of a "light ray" as its only 3D inductive bias.
4.  **Superior Performance:** Surprisingly, `RayZer` often performs better than "Oracle" models (models given the ground truth). This is because the "ground truth" (often from `COLMAP`) is sometimes lower quality than what `RayZer` learns to estimate for itself.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Novel View Synthesis (NVS)
`Novel View Synthesis` is the task of taking a few images of an object or scene and generating a brand-new image from a camera angle that was never captured. Imagine taking three photos of a chair and then using a computer to "walk around" the chair and see the back, even though you never photographed the back.

### 3.1.2. Transformers and Self-Attention
A `Transformer` is a neural network architecture that relies on the `Self-Attention` mechanism. In vision, it treats an image like a sequence of "tokens" (small patches).
The core `Attention` formula is:
\$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\$
- $Q$ (Query): What the token is looking for.
- $K$ (Key): What the token contains.
- $V$ (Value): The actual information to be passed along.
- $d_k$: The dimension of the keys, used for scaling.

### 3.1.3. Camera Poses and SE(3)
To understand a 3D scene, we must know the camera's `Pose`. This is usually represented in $SE(3)$ (Special Euclidean group in 3D), which consists of:
- **Rotation ($R$):** A $3 \times 3$ matrix showing where the camera is pointing.
- **Translation ($t$):** A `3`-element vector showing where the camera is located in space.

### 3.1.4. Plücker Rays
Instead of just using `(x, y)` coordinates for pixels, 3D models often use `Rays`. A `Plücker Ray` is a 6D vector that defines a line in 3D space. It consists of a direction vector and a moment vector. This helps the model understand the geometry of how light travels from the scene to the camera lens.

## 3.2. Technological Evolution
1.  **Optimization-based (e.g., NeRF):** Required hours of training for a *single* scene.
2.  **Supervised Feed-forward (e.g., LRM, LVSM):** Can reconstruct a scene in a fraction of a second but require massive datasets with perfect camera labels.
3.  **Self-supervised (RayZer):** The next step—removing the need for those labels entirely, allowing the model to learn from raw internet videos.

## 3.3. Differentiation Analysis
Unlike previous "Pose-free" models like `RUST`, which estimate the scene first and then the pose, `RayZer` uses a **Pose-first** approach. It predicts the camera parameters first and uses them to guide the scene reconstruction. The authors argue that even a "noisy" camera prediction is a better starting point for reconstruction than no camera information at all.

---

# 4. Methodology

## 4.1. Principles
The core principle of `RayZer` is **3D-aware image auto-encoding**.
- **Encoder:** Takes a set of images and breaks them down into two parts: a "Map" of the scene and a set of "Camera Poses."
- **Decoder:** Takes that Map and a Pose and "renders" (draws) the image back.
- **Learning:** If the rendered image matches the original image, the model's understanding of both the scene and the camera must be correct.

## 4.2. Core Methodology In-depth (Layer by Layer)

### Step 1: Image Tokenization
The model starts by taking $K$ input images $\mathcal{T} = \{I_1, I_2, ..., I_K\}$. Each image is "patchified" into small squares of size $s \times s$. These patches are flattened and projected into a latent vector space of dimension $d$. 
To keep track of where these patches came from, the model adds **Positional Embeddings**:
1.  **Spatial Embedding:** Tells the model where the patch is *inside* the image.
2.  **Image Index Embedding:** Tells the model *which* image the patch belongs to. This is crucial for video frames to maintain temporal order.

### Step 2: Camera Estimation
The model uses a transformer-based `Camera Estimator` ($\mathcal{E}_{cam}$). It takes the image tokens and a set of learnable "camera tokens" $\mathbf{p}$. Through layers of self-attention, the tokens interact:
\$
\{\mathbf{f}^*, \mathbf{p}^*\} = \mathcal{E}_{cam}(\{\mathbf{f}, \mathbf{p}\})
\$
where $\mathbf{f}^*$ are updated image features and $\mathbf{p}^*$ are updated camera features.

From the updated camera tokens $\mathbf{p}^*$, the model predicts:
1.  **Relative Pose ($p_i$):** One view is chosen as the "anchor" (canonical view). For all other views, the model predicts the rotation and translation relative to that anchor using a Multi-Layer Perceptron (MLP):
    \$
    p_i = \mathbf{MLP}_{pose}([\mathbf{p}_i^*, \mathbf{p}_c^*])
    \$
    Here, $[\cdot, \cdot]$ means joining the two vectors together. The output $p_i$ is a 9D vector (6D for rotation, 3D for translation).
2.  **Intrinsics (Focal Length):** The model assumes all images share the same camera lens properties (focal length):
    \$
    \mathrm{focal} = \mathbf{MLP}_{focal}(\mathbf{p}_c^*)
    \$

### Step 3: Ray Map Construction
The predicted poses and focal length are used to generate **Plücker Ray Maps** ($\mathcal{R}$). For every pixel in every image, the model calculates a 6D ray vector that represents the physical path of light for that pixel.

### Step 4: Scene Reconstruction
Next, the model builds a 3D representation of the scene. It uses a `Scene Reconstructor` ($\mathcal{E}_{scene}$). To avoid "cheating," the model only looks at a subset of images $\mathcal{T}_A$ to build the scene.
It fuses the image tokens $\mathbf{f}_A$ with their corresponding ray tokens $\mathbf{r}_A$:
\$
\mathbf{x}_A = \mathbf{MLP}_{fuse}([\mathbf{f}_A, \mathbf{r}_A])
\$
Then, it updates a set of "latent scene tokens" $\mathbf{z}$:
\$
\{\mathbf{z}^*, \mathbf{x}_A^*\} = \mathcal{E}_{scene}(\{\mathbf{z}, \mathbf{x}_A\})
\$
$\mathbf{z}^*$ is the final compressed representation of the 3D scene.

### Step 5: Rendering Decoder
To produce a new image, the `Rendering Decoder` ($\mathcal{D}_{render}$) takes the scene tokens $\mathbf{z}^*$ and the ray map of a "target" camera view $\mathbf{r}_B$:
\$
\{\mathbf{r}^*, \mathbf{z}'\} = \mathcal{D}_{render}(\{\mathbf{r}, \mathbf{z}^*\})
\$
Finally, an RGB MLP converts these tokens back into actual pixel colors:
\$
\hat{I} = \mathbf{MLP}_{rgb}(\mathbf{r}^*)
\$

### Step 6: Self-Supervised Loss
The model is trained by comparing the rendered image $\hat{I}$ to the actual image from the dataset $I$. The loss function $\mathcal{L}$ is:
\$
\mathcal{L} = \frac{1}{K_B} \sum_{\hat{I} \in \hat{\mathcal{T}}_B} (\mathtt{MSE}(I, \hat{I}) + \lambda \cdot \mathtt{Percep}(I, \hat{I}))
\$
- **MSE (Mean Squared Error):** Measures the average squared difference between pixel colors. 
  \$
  \mathrm{MSE} = \frac{1}{n} \sum (I - \hat{I})^2
  \$
- **Percep (Perceptual Loss):** Uses a pre-trained brain-like network to see if the images "look" similar to a human, even if individual pixels are slightly off.
- $\lambda$: A weight (set to 0.2) to balance these two goals.

  The following figure (Figure 3 from the original paper) illustrates this entire data flow:

  ![Figure 3. RayZer self-supervised learning framework.RayZer takes inunposed and uncalibratedmulti-viewage $\\mathcal { T }$ and predicts poses $\\mathcal { P }$ of all views. The predicted cameras are then converted into pixel-aligned Plücker ray maps $\\mathcal { R }$ . (Middle) RayZer uses a subset of input images, $\\mathcal { T } _ { A }$ , as well as their previously predicted camera Plücker ray maps, $\\mathcal { R } _ { A }$ , to predict a latent scene representation. Here, the Plücker ray maps, $\\mathcal { R } _ { A }$ , rv n efecivndoreotucRih)RayZera endetar a ivenh representation $\\mathbf { z } ^ { \\ast }$ and a target camera. During training, we use $\\mathcal { R } _ { B }$ , which is the previously predicted cameras Plücker ray maps of $\\mathcal { T } _ { B }$ , to render $\\hat { \\mathcal { T } } _ { B }$ This allows training RayZer end-to-end with self-supervised photometric losses between inputs $\\mathcal { T } _ { B }$ and their renderings $\\hat { \\mathcal { T } } _ { B }$ .](images/3.jpg)
  *该图像是示意图，展示了RayZer自监督学习框架的各个部分，包括相机估计、场景重建和渲染过程。文中通过 `L = rac{1}{|g|} extstyle orall_{i_j eq eta} (MSE(I, ilde{I}) + au imes ext{Percep}(I, ilde{I}))` 表示损失函数。*

---

# 5. Experimental Setup

## 5.1. Datasets
1.  **DL3DV:** A large-scale dataset of real-world indoor and outdoor scenes.
2.  **RealEstate10k:** A dataset of thousands of video clips from YouTube house tours.
3.  **Objaverse:** A massive collection of 3D objects (rendered into videos for this paper).

## 5.2. Evaluation Metrics

### 5.2.1. PSNR (Peak Signal-to-Noise Ratio)
*   **Conceptual Definition:** Measures the ratio between the maximum possible power of a signal and the power of corrupting noise. In images, higher PSNR means the reconstruction is higher quality and less "noisy."
*   **Mathematical Formula:**
    \$
    \mathrm{PSNR} = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{\mathrm{MSE}} \right)
    \$
*   **Symbol Explanation:** $MAX_I$ is the maximum possible pixel value (usually 255 or 1.0). $\mathrm{MSE}$ is the Mean Squared Error between the images.

### 5.2.2. SSIM (Structural Similarity Index Measure)
*   **Conceptual Definition:** Unlike MSE, which just looks at pixel values, SSIM compares local patterns of pixel intensities that have been normalized for luminance and contrast. It better reflects human perception of "structure."
*   **Mathematical Formula:**
    \$
    \mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
    \$
*   **Symbol Explanation:** $\mu$ is the average, $\sigma^2$ is the variance, and $\sigma_{xy}$ is the covariance of the image patches $x$ and $y$. $c_1, c_2$ are constants to stabilize the division.

### 5.2.3. LPIPS (Learned Perceptual Image Patch Similarity)
*   **Conceptual Definition:** Measures how "different" two images look using features from a deep neural network. Lower scores mean the images are more perceptually similar.
*   **Mathematical Formula:** Calculated as the distance between activations of a pre-trained network (like VGG or AlexNet).

## 5.3. Baselines
The authors compare `RayZer` against:
- **GS-LRM / LVSM:** "Oracle" methods. They are given the "Ground Truth" camera poses during training and testing.
- **PF-LRM:** A supervised method that learns to predict poses but requires pose labels during its own training.

  ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The most striking result is found in the real-world datasets (`DL3DV` and `RealEstate`). `RayZer` (unsupervised) actually **outperforms** the oracle models that were given the labels.
The authors explain this through a "noisy label" hypothesis: The labels provided by `COLMAP` are not perfect. When a model like `LVSM` is forced to learn from these imperfect labels, it hits a performance ceiling. `RayZer`, by learning its own pose space that best fits the image data, avoids this limitation.

## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper, showing performance on the `DL3DV` dataset:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Training Supervision</th>
<th rowspan="2">Inference w. COLMAP Cam.</th>
<th colspan="3">Even Sample</th>
<th colspan="3">Random Sample</th>
</tr>
<tr>
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
<td colspan="9"><i>"Oracle" methods (assume inputs are posed & use pose annotations during training)</i></td>
</tr>
<tr>
<td>GS-LRM</td>
<td>2D + Camera</td>
<td>Yes</td>
<td>23.49</td>
<td>0.712</td>
<td>0.252</td>
<td>23.02</td>
<td>0.705</td>
<td>0.266</td>
</tr>
<tr>
<td>LVSM</td>
<td>2D + Camera</td>
<td>Yes</td>
<td>23.69</td>
<td>0.723</td>
<td>0.242</td>
<td>23.10</td>
<td>0.703</td>
<td>0.257</td>
</tr>
<tr>
<td colspan="9"><i>Unsupervised methods (inputs are un-posed & no pose annotations used during training)</i></td>
</tr>
<tr>
<td><b>RayZer (Ours)</b></td>
<td><b>2D</b></td>
<td><b>No</b></td>
<td><b>24.36</b></td>
<td><b>0.757</b></td>
<td><b>0.209</b></td>
<td><b>23.72</b></td>
<td><b>0.733</b></td>
<td><b>0.222</b></td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies
The authors tested several variations to prove their design choices:
- **Latent vs. Explicit (3DGS):** Using an explicit 3D representation like `Gaussian Splatting` without labels failed to converge. The `Latent Set` representation is more flexible and "learnable."
- **Ray Prior:** Removing the `Plücker Ray` maps and just using raw camera numbers significantly dropped performance, proving that the physical "ray" concept is a powerful guide.
- **Pose First vs. Scene First:** Predicting the scene before the pose (like the model `RUST` does) resulted in much worse performance (PSNR dropped from 24.36 to 13.31).

  ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
`RayZer` represents a significant milestone in 3D Vision. It proves that we can train large-scale, high-performance view synthesis models using **only 2D images**. By abandoning the reliance on noisy, hard-to-get camera labels, `RayZer` opens the door to using the nearly infinite supply of unlabeled video data on the web to build foundational 3D models.

## 7.2. Limitations & Future Work
- **Fine Details:** The authors noted that `RayZer` still struggles with extremely fine geometry (like thin plant leaves) and complex reflections/occlusions.
- **Pose Space Disentanglement:** While the model predicts "poses," these poses exist in a "learned space" that doesn't perfectly align with real-world coordinates. Finding a way to align these without supervision is a future challenge.

## 7.3. Personal Insights & Critique
The most profound takeaway from this paper is the **"Noisy Label" realization**. In many fields of AI, we assume "Ground Truth" is the gold standard. This paper shows that when Ground Truth is generated by another algorithm (like `COLMAP`), it might actually be *holding back* the next generation of models. 

**Critique:** While the model is "self-supervised," it still requires a specific data structure (multi-view images or video frames). It cannot yet learn from a completely random collection of single images of different objects. However, given that video is the most common form of visual data today, this is a very practical and powerful limitation to overcome.