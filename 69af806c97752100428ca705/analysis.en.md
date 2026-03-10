# 1. Bibliographic Information

## 1.1. Title
The title of the paper is **"MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos"**. This title indicates the system's name (MegaSaM) and its core capabilities: accuracy, speed, and robustness in estimating structure (3D geometry) and motion (camera parameters) from videos that are "casual" (uncontrolled, handheld) and "dynamic" (containing moving objects).

## 1.2. Authors
The authors are **Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely**.
Their affiliations include:
*   **Google DeepMind**
*   **UC Berkeley** (University of California, Berkeley)
*   **University of Michigan**

    This collaboration suggests a strong blend of industrial research resources (Google DeepMind) and academic expertise in computer vision and graphics (Berkeley, Michigan).

## 1.3. Journal/Conference
The paper was published as a **preprint on arXiv**.
*   **Publication Status:** Preprint (as of the provided text date).
*   **Venue Reputation:** arXiv is the primary repository for preprints in computer science and physics. While not a peer-reviewed conference proceedings itself, papers published here often target top-tier venues like CVPR, ICCV, or ECCV. The affiliation with Google DeepMind and UC Berkeley suggests high-quality research intended for major computer vision conferences.

## 1.4. Publication Year
The paper was published on **December 5, 2024** (UTC). This places it as very recent work in the field of computer vision.

## 1.5. Abstract
The paper presents **MegaSaM**, a system for estimating camera parameters and depth maps from casual monocular videos of dynamic scenes.
*   **Objective:** To overcome the limitations of conventional Structure from Motion (SfM) and Simultaneous Localization and Mapping (SLAM) techniques, which typically assume static scenes and large camera parallax (movement).
*   **Methodology:** The authors extend a deep visual SLAM framework (specifically DROID-SLAM) with careful modifications to training and inference. Key innovations include integrating monocular depth priors, motion probability maps, and an uncertainty-aware global Bundle Adjustment (BA) scheme.
*   **Results:** Extensive experiments on synthetic and real videos show the system is significantly more accurate and robust than prior work in camera pose and depth estimation, with faster or comparable running times.
*   **Conclusion:** The system scales to real-world videos with complex dynamics and unconstrained camera paths, including those with little camera parallax.

## 1.6. Original Source Link
*   **Original Source Link:** [https://arxiv.org/abs/2412.04463](https://arxiv.org/abs/2412.04463)
*   **PDF Link:** [https://arxiv.org/pdf/2412.04463v2](https://arxiv.org/pdf/2412.04463v2)
*   **Project Page:** [https://mega-sam.github.io/](https://mega-sam.github.io/)

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem addressed is **camera tracking and 3D reconstruction from casual monocular videos**.
*   **The Challenge:** Traditional SfM and SLAM algorithms work well for stationary scenes with significant camera movement (parallax). However, they fail on "casual" videos, which are often handheld, have limited camera movement (e.g., mostly rotational), unknown focal lengths, and contain moving objects (dynamic scenes).
*   **Gaps in Prior Research:**
    *   Conventional methods produce erroneous estimates when scene dynamics or limited parallax are present.
    *   Recent neural network-based approaches attempt to solve this but are either computationally expensive (slow) or brittle (unstable) when faced with uncontrolled camera motion or unknown fields of view.
*   **Innovative Idea:** The authors reexamine deep visual SLAM frameworks. They hypothesize that with specific modifications to training (handling dynamics) and inference (handling uncertainty), these frameworks can be made robust enough for in-the-wild dynamic videos without needing expensive test-time fine-tuning.

## 2.2. Main Contributions / Findings
The paper makes several primary contributions:
1.  **MegaSaM Pipeline:** A full system for accurate, fast, and robust camera tracking and depth estimation from dynamic monocular videos.
2.  **Deep Visual SLAM Extension:** They demonstrate that a learned differentiable Bundle Adjustment (BA) layer (from DROID-SLAM) is critical for dynamic videos. They extend it by integrating **monocular depth priors** and **motion probability maps**.
3.  **Uncertainty-Aware Global BA:** A novel scheme that analyzes the observability of structure and camera parameters. It uses epistemic uncertainty to decide when to apply mono-depth regularization, improving robustness when camera parameters are poorly constrained (e.g., rotational motion).
4.  **Consistent Video Depth:** A method to obtain accurate, consistent video depths efficiently without test-time network fine-tuning.
5.  **Performance:** The system significantly outperforms prior and concurrent baselines (like CasualSAM, MonST3R) in accuracy and robustness while maintaining competitive runtime.

    ![Figure 1. MegaSaM enables accurate, fast and robust estimation of cameras and scene structure from a casually captured monocular video of a dynamic scene. Top: input video frames (every tenth frame shown). Bottom: our estimated camera and 3D point clouds unprojected by predicted video depths without any postprocessing.](images/1.jpg)
    *Figure 1 from the original paper illustrates the system's capability. The top row shows input video frames from a casual dynamic scene. The bottom row shows the estimated camera trajectory and 3D point clouds unprojected from the predicted depths, demonstrating accurate reconstruction without post-processing.*

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, several foundational computer vision concepts must be clarified:

*   **Structure from Motion (SfM):** A technique for estimating 3D structure (point cloud) and camera motion from a set of 2D images. It typically relies on finding correspondences (matching points) across images.
*   **Simultaneous Localization and Mapping (SLAM):** Similar to SfM but often operates in real-time on video sequences. It estimates the camera's position (localization) and the map of the environment (mapping) simultaneously.
*   **Bundle Adjustment (BA):** An optimization technique used in SfM/SLAM to refine the 3D coordinates of scene points and camera parameters simultaneously. It minimizes the **reprojection error**, which is the difference between observed image points and projected 3D points.
*   **Monocular Depth:** Estimating depth (distance from camera) from a single image. This is an ill-posed problem (ambiguous without extra info), so modern methods use deep learning priors.
*   **Parallax:** The apparent displacement of an object when viewed from different positions. Large parallax (significant camera translation) is crucial for traditional SfM to triangulate depth. Casual videos often lack this.
*   **Epistemic Uncertainty:** Uncertainty arising from the model's lack of knowledge or data (e.g., when geometry is unobservable due to lack of camera motion). This is distinct from aleatoric uncertainty (noise in data).
*   **SE(3):** The Special Euclidean group in 3 dimensions. It represents rigid body transformations (rotation and translation), used to describe camera poses.

## 3.2. Previous Works
The paper builds upon and compares against several key prior studies:

*   **DROID-SLAM [59]:** A deep visual SLAM system that uses a differentiable BA layer. It iteratively updates scene geometry and camera poses. MegaSaM extends this framework.
*   **CasualSAM [78]:** Estimates camera parameters and dense depth from dynamic videos by fine-tuning mono-depth networks. MegaSaM improves upon this by avoiding expensive fine-tuning.
*   **Particle-SfM [79] & LEAP-VO [6]:** These methods infer moving object masks based on trajectories to downweight features during BA. MegaSaM integrates motion probability maps directly into the SLAM paradigm.
*   **MonST3R [76]:** A concurrent work using 3D point cloud representations (from DuST3R) for dynamic scenes. MegaSaM shows superior accuracy and robustness.
*   **DepthAnything [71] & UniDepth [43]:** Off-the-shelf monocular depth models used by MegaSaM for initialization and priors.

## 3.3. Technological Evolution
The field has evolved from conventional feature-based SfM (e.g., COLMAP, ORB-SLAM) which struggles with dynamics, to learning-based methods.
1.  **Era 1: Static Assumption.** Traditional SfM assumed static scenes.
2.  **Era 2: Dynamic Handling via Masking.** Methods started segmenting moving objects to ignore them.
3.  **Era 3: Neural Priors.** Recent works use neural networks to predict depth or flow to aid reconstruction.
4.  **Era 4 (MegaSaM):** Integrating neural priors *into* the optimization loop (differentiable BA) with uncertainty handling, allowing for robust performance on unconstrained videos without per-video fine-tuning.

## 3.4. Differentiation Analysis
MegaSaM differentiates itself from main methods in related work through:
*   **Vs. Conventional SfM:** It does not rely on static scene assumptions or large baselines.
*   **Vs. CasualSAM:** It does not require test-time network fine-tuning, making it faster.
*   **Vs. DROID-SLAM:** It explicitly handles dynamic objects via learned motion maps and handles unknown focal lengths/limited parallax via uncertainty-aware regularization.
*   **Vs. MonST3R:** It uses a differentiable SLAM framework rather than a global point cloud alignment, showing better trajectory accuracy in experiments.

# 4. Methodology

## 4.1. Principles
The core principle of MegaSaM is to combine the iterative refinement capability of **deep visual SLAM** with **monocular depth priors** and **uncertainty quantification**.
*   **Intuition:** Standard SLAM fails on dynamic videos because moving objects violate the static scene assumption. Standard mono-depth is temporally inconsistent. By combining them in a differentiable optimization loop, the system can use the mono-depth to constrain the solution when visual cues (parallax) are weak, and use the visual cues to correct the mono-depth when they are strong.
*   **Theoretical Basis:** The system minimizes a reprojection error cost function using a differentiable approximation of the Levenberg-Marquardt algorithm. It treats object motion as uncertainty to be downweighted during optimization.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Deep Visual SLAM Formulation
MegaSaM builds on DROID-SLAM. The system maintains state variables: per-frame low-resolution disparity maps $\hat { \mathbf { d } } _ { i }$ and camera poses $\hat { \bf G } _ { i } \in S E ( 3 )$.
These variables are updated iteratively via a differentiable BA layer over a frame-graph of image pairs `( I _ { i } , I _ { j } )`.

**Correspondence Prediction:**
Given frames `I _ { i }` and `I _ { j }`, the model predicts a 2D correspondence field $\hat { \mathbf { u } } _ { i j }$ and confidence $\hat { \mathbf { w } } _ { i j }$ iteratively:
\$
\big ( \hat { \mathbf { u } } _ { i j } ^ { k + 1 } , \hat { \mathbf { w } } _ { i j } ^ { k + 1 } \big ) = F ( I _ { i } , I _ { j } , \hat { \mathbf { u } } _ { i j } ^ { k } , \hat { \mathbf { w } } _ { i j } ^ { k } )
\$
where $k$ denotes the iteration.

**Rigid-Motion Constraint:**
The rigid-motion correspondence field is derived from camera egomotion and disparity:
\$
{ \bf u } _ { i j } = \pi \left( \hat { \bf G } _ { i j } \circ \pi ^ { - 1 } ( { \bf p } _ { i } , \hat { \bf d } _ { i } , K ^ { - 1 } ) , K \right)
\$
*   ${ \bf p } _ { i }$: Grid of pixel coordinates.
*   $\pi$: Projection function.
*   $\hat { \mathbf { G } } _ { i j } = \hat { \mathbf { G } } _ { j } \circ \hat { \mathbf { G } } _ { i } ^ { - 1 }$: Relative camera pose.
*   $K$: Camera intrinsic matrix.
*   $\hat { \bf d } _ { i }$: Disparity map.

**Differentiable Bundle Adjustment:**
The system optimizes poses, focal length $\hat { f }$, and disparity by minimizing weighted reprojection cost:
\$
\mathcal { C } ( \hat { \mathbf { G } } , \hat { \mathbf { d } } , \hat { f } ) = \sum _ { ( i , j ) \in \mathcal { P } } | | \hat { \mathbf { u } } _ { i j } - \mathbf { u } _ { i j } | | _ { \Sigma _ { i j } } ^ { 2 }
\$
*   $\Sigma _ { i j } = \mathrm { d i a g } ( \hat { \mathbf { w } } _ { i j } ) ^ { - 1 }$: Weights based on confidence.
*   $\mathcal { P }$: Set of frame pairs.

    Optimization is performed using the Levenberg-Marquardt algorithm:
\$
\left( \mathbf { J } ^ { T } \mathbf { W } \mathbf { J } + \lambda \mathrm { d i a g } ( \mathbf { J } ^ { T } \mathbf { W } \mathbf { J } ) \right) \boldsymbol { \Delta } \boldsymbol { \xi } = \mathbf { J } ^ { T } \mathbf { W } \mathbf { r }
\$
*   $\Delta \pmb { \xi } = ( \Delta \mathbf G , \Delta \mathbf d , \Delta f ) ^ { T }$: Parameter updates.
*   $\mathbf { J }$: Jacobian of reprojection residuals.
*   $\mathbf { W }$: Diagonal matrix of weights $\hat { \mathbf { w } } _ { i j }$.
*   $\lambda$: Damping factor.

    To solve this efficiently, the Hessian is divided into block matrices (Schur complement trick):
\$
\begin{array} { r } { \left[ \mathbf { H } _ { \mathbf { G } , f } \quad \mathbf { E } _ { \mathbf { G } , f } \right] \left[ \Delta \xi _ { \mathbf { G } , f } \right] = \binom { \tilde { r } _ { \mathbf { G } , f } } { \tilde { r } _ { \mathbf { d } } } } \\ { \mathbf { E } _ { \mathbf { G } , f } ^ { T } \quad \mathbf { H } _ { \mathbf { d } } } \end{array}
\$
This leads to the fully differentiable BA update:
\$
\begin{array} { r l r } & { \Delta \pmb { \xi } _ { \mathbf { G } , f } = \left[ \mathbf { H } _ { \mathbf { G } , f } - \mathbf { E } _ { \mathbf { G } , f } \mathbf { H } _ { \mathbf { d } } ^ { - 1 } \mathbf { E } _ { \mathbf { G } , f } { } ^ { T } \right] ^ { - 1 } \left( \tilde { r } _ { \mathbf { G } , f } - \mathbf { E } _ { \mathbf { G } , f } \mathbf { H } _ { \mathbf { d } } ^ { - 1 } \tilde { r } _ { \mathbf { d } } \right) } & \\ & { \qquad ( 5 ) } & \\ & { \Delta \mathbf { z } = \mathbf { H } _ { \mathbf { d } } ^ { - 1 } ( \tilde { r } _ { \mathbf { d } } - \mathbf { E } _ { \mathbf { G } , f } ^ { T } \Delta \pmb { \xi } _ { \mathbf { G } , f } ) } & { ( 6 ) } \end{array}
\$

**Training:**
The flow and uncertainty predictions are trained end-to-end on static scenes:
\$
\mathcal { L } _ { \mathrm { s t a t i c } } = \mathcal { L } _ { \mathrm { c a m } } + w _ { \mathrm { f l o w } } \mathcal { L } _ { \mathrm { f l o w } }
\$
*   $\mathcal { L } _ { \mathrm { c a m } }$: Loss comparing estimated camera parameters to ground truth.
*   $\mathcal { L } _ { \mathrm { f l o w } }$: Loss comparing ego-motion induced flows to ground truth.

    ![Figure 10. Architecture of fow, confidence and movement map predictor. The gray blocks belong to the nework $F$ for flow and confidence prediction, and the blue blocks belong to the network `F _ { m }` for object movement map prediction. In the first stage, we perform ego-motion pretraining for $F$ . In the second stage, we perform dynamic fine-tuning for `F _ { m }` while fixing the parameters of $F$ .](images/10.jpg)
    *该图像是示意图，展示了流量、置信度和运动图预测器的架构。灰色块表示用于流量和置信度预测的网络 $F$，蓝色块则表示用于对象运动图预测的网络 $F_m$。第一阶段进行 $F$ 的自我运动预训练，第二阶段在固定 $F$ 参数的情况下，对 $F_m$ 进行动态微调。*
    *Figure 10 from the original paper shows the architecture of the flow, confidence, and movement map predictors. The gray blocks represent network $F$ (flow/confidence), and blue blocks represent network $F_m$ (motion map). Training happens in two stages: ego-motion pretraining for $F$, then dynamic fine-tuning for $F_m$.*

### 4.2.2. Scaling to Dynamic Videos
To handle dynamics and limited parallax, MegaSaM introduces key modifications.

**Learning Motion Probability:**
Instead of relying solely on pairwise uncertainty, an additional network `F _ { m }` predicts an object movement probability map $\mathbf { m } _ { i }$:
\$
\mathbf { m } _ { i } \in \mathcal { R } ^ { \frac { H } { 8 } \times \frac { w } { 8 } } = F _ { m } \left( \{ I _ { i } \} \cup \mathcal { N } ( i ) \right)
\$
*   $\mathcal { N } ( i )$: Set of neighboring keyframes.
*   $\mathbf { m } _ { i }$: Predicts pixels corresponding to dynamic content.

    During BA, the final weights are combined: $\tilde { \mathbf { w } } _ { i j } = \hat { \mathbf { w } } _ { i j } \mathbf { m } _ { i }$. This downweights dynamic elements.

**Two-Stage Training:**
1.  **Ego-motion Pretraining:** Train $F$ on static scenes (Eq. 7).
2.  **Dynamic Fine-tuning:** Freeze $F$, finetune `F _ { m }` on dynamic videos. Loss:
    \$
    \mathcal { L } _ { \mathrm { d y n a m i c } } = \mathcal { L } _ { \mathrm { c a m } } + w _ { \mathrm { m o t i o n } } \mathcal { L } _ { \mathrm { C E } }
    \$
    *   $\mathcal { L } _ { \mathrm { C E } }$: Cross-entropy loss for motion prediction.

**Disparity and Camera Initialization:**
Instead of constant initialization, disparity $\hat { \mathbf { d } }$ is initialized with disparity from **DepthAnything [71]**. Camera focal length is initialized using **UniDepth [43]** predictions.

**Inference Pipeline:**
1.  **Frontend:** Registers cameras for keyframes via sliding window BA. Cost function includes mono-depth regularization:
    \$
    \mathcal { C } = \sum _ { ( i , j ) \in \mathcal { P } } | | \widehat { \mathbf { u } } _ { i j } - \mathbf { u } _ { i j } | | _ { \Sigma _ { i j } } ^ { 2 } + w _ { d } \sum _ { i } | | \widehat { \mathbf { d } } _ { i } - D _ { i } ^ { \mathrm { a l i g n } } | | ^ { 2 } .
    \$
    *   $D _ { i } ^ { \mathrm { a l i g n } }$: Aligned monocular disparity.
    *   `w _ { d }`: Regularization weight.

2.  **Uncertainty-Aware Global BA:**
    The backend performs global BA. To decide when to use mono-depth regularization, the system estimates epistemic uncertainty $\Sigma _ { \theta }$ via the inverse Hessian:
    \$
    \Sigma _ { \theta } \approx \mathrm { d i a g } \left( - \mathbf { H } ( \theta ^ { * } ) \right) ^ { - 1 }
    \$
    *   $\mathbf { H } ( \theta ^ { * } )$: Hessian at the MAP estimate.
    *   High uncertainty implies parameters are unobservable (e.g., static camera).

        The mono-depth regularization weight is set based on median disparity Hessian:
    \$
    w _ { d } = \gamma _ { d } \exp \left( - \beta _ { d } \mathrm { m e d } \left( \mathrm { d i a g } ( \mathbf { H _ { d } } ) \right) \right)
    \$
    *   If uncertainty is high (Hessian diagonal is small), $w_d$ increases, relying more on the depth prior.
    *   Focal length optimization is disabled if $H _ { f } < \tau _ { f }$.

        ![Figure 4. Visualization of epistemic uncertainty. From left to right, we visualize camera paths, reference image and corresponding epistemic uncertainty of disparity. The geometry is not observable from the top example with little camera parallax, as indicated by the larger uncertainty. The peak on the bottom uncertainty map corresponds to the epipole for forward moving motion.](images/4.jpg)
        *Figure 4 from the original paper visualizes epistemic uncertainty. The top example (rotational motion) shows high uncertainty (geometry unobservable), while the bottom (forward motion) shows lower uncertainty. This guides the system to apply regularization where needed.*

        ![Figure 3. Learned movement map. Left: input video frame, right: corresponding learned motion probability map.](images/3.jpg)
        *Figure 3 from the original paper shows the learned movement map. The right image highlights dynamic regions (people walking), which the system learns to downweight during camera tracking.*

### 4.2.3. Consistent Depth Optimization
After camera estimation, video depths are refined at higher resolution without fine-tuning the network. The objective is:
\$
\mathcal { C } _ { \mathrm { c v d } } = w _ { \mathrm { f l o w } } \mathcal { C } _ { \mathrm { f l o w } } + w _ { \mathrm { t e m p } } \mathcal { C } _ { \mathrm { t e m p } } + w _ { \mathrm { p r i o r } } \mathcal { C } _ { \mathrm { p r i o r } }
\$
*   $\mathcal { C } _ { \mathrm { f l o w } }$: Pairwise 2D flow reprojection loss.
*   $\mathcal { C } _ { \mathrm { t e m p } }$: Temporal depth consistency loss.
*   $\mathcal { C } _ { \mathrm { p r i o r } }$: Scale invariant mono-depth prior loss.

**Flow Loss (Appendix A.3):**
\$
\mathcal { C } _ { \mathrm { f l o w } } ^ { i  j } = \hat { M } _ { i } | | \mathbf { u } _ { i j } - \mathbf { p } _ { i } , \mathrm { H o w } _ { i  j } ( \mathbf { p } _ { i } ) | | _ { 1 } + \log ( \frac { 1 } { \hat { M } _ { i } } ) ,
\$
*   $\hat { M } _ { i }$: Aleatoric uncertainty map.
*   $\mathrm { H o w } _ { i  j }$: Optical flow from off-the-shelf estimator.

**Temporal Loss:**
\$
\begin{array} { r l r } & { \mathcal { C } _ { \mathrm { t e m p } } ^ { i  j } = \hat { M } _ { i } \delta ( \mathbf { P } _ { z } ^ { i  j } , \hat { D } _ { j } ( \mathbf { p } + \mathrm { f l o w } _ { i  j } ( \mathbf { p } ) ) ) + \log ( \frac { 1 } { \hat { M } _ { i } } ) } & \\ & { \delta ( a , b ) = | | \operatorname* { m a x } ( \frac { a } { b } , \frac { b } { a } ) | | _ { 1 } } & \\ & { \mathbf { P } _ { z } ^ { i  j } = ( D _ { i } ( \mathbf { p } ) \mathbf { R } _ { i  j } \mathbf { K } ^ { - 1 } \mathbf { p } + \mathbf { t } _ { i  j } ) _ { [ z ] } } & { ( 1 5 ) } \end{array}
\$
*   Encourages depth consistency along the optical flow.

**Prior Loss:**
\$
\mathcal { C } _ { \mathrm { p r i o r } } = \mathcal { C } _ { \mathrm { s i } } + w _ { \mathrm { g r a d } } \mathcal { C } _ { \mathrm { g r a d } } + w _ { \mathrm { n o r m a l } } \mathcal { C } _ { \mathrm { n o r m a l } }
\$
Includes scale-invariant depth loss $\mathcal { C } _ { \mathrm { s i } }$, gradient matching $\mathcal { C } _ { \mathrm { g r a d } }$, and surface normal loss $\mathcal { C } _ { \mathrm { n o r m a l } }$.

![该图像是示意图，展示了基于视频的摄像机跟踪和一致性视频深度估计的过程。左侧部分显示了输入视频和可微分BA的架构，右侧部分展示了深度、不确定性和流的更新过程。](images/9.jpg)
*Figure 9 from the original paper (Appendix) illustrates the system overview. Left: Camera tracking with differentiable BA and mono-depth initialization. Right: Consistent video depth estimation minimizing flow and depth losses.*

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize three main benchmarks:
1.  **MPI Sintel [4]:** Animated video sequences with complex object motions and camera paths. Contains 18 sequences, each with 20-50 images. Used for quantitative evaluation.
2.  **DyCheck [12]:** Real-world videos of dynamic scenes from handheld cameras. Each video has 180-500 frames. Ground truth provided by Shape of Motion [64].
3.  **In-the-wild:** 12 dynamic videos from DynIBaR [32]. Features long duration (100-600 frames) and uncontrolled camera paths. Ground truth cameras obtained via COLMAP after masking moving objects.

    **Data Example:** The Sintel dataset includes scenes like "alley 1", "ambush", "market", etc., which feature significant dynamic characters and camera movement, making them ideal for testing robustness.

## 5.2. Evaluation Metrics
The paper uses standard metrics for camera pose and depth.

**Camera Pose Metrics:**
1.  **Absolute Translation Error (ATE):** Measures the absolute difference between estimated and ground truth camera positions.
    *   *Formula:* $ATE = \frac{1}{N} \sum_{i=1}^{N} || \mathbf{t}_i^{gt} - \mathbf{t}_i^{est} ||$
    *   *Symbols:* $\mathbf{t}_i$: Translation vector at frame $i$. $N$: Number of frames.
2.  **Relative Translation Error (RTE):** Measures translation error over fixed intervals.
3.  **Relative Rotation Error (RRE):** Measures rotation error over fixed intervals.
    *   *Note:* Trajectories are normalized to unit length to account for scale differences.

**Depth Metrics:**
1.  **Absolute Relative Error (abs-rel):** Mean of absolute relative difference.
    *   *Formula:* $\frac{1}{N} \sum \frac{|D^{gt} - D^{est}|}{D^{gt}}$
2.  **log RMSE:** Root mean squared error in log space.
3.  **$\delta _ { 1 . 2 5 }$:** Percentage of pixels where predicted depth is within a factor of 1.25 of ground truth.
    *   *Formula:* $\% \text{ where } \max(\frac{D^{gt}}{D^{est}}, \frac{D^{est}}{D^{gt}}) < 1.25$

## 5.3. Baselines
MegaSaM is compared against:
*   **ACE-Zero [3]:** Scene coordinate regression for static scenes.
*   **CasualSAM [78]:** Joint camera/depth optimization via mono-depth fine-tuning.
*   **RoDynRF [34]:** Dynamic radiance fields.
*   **Particle-SfM [79] & LEAP-VO [6]:** Motion segmentation based VO/SfM.
*   **MonST3R [76]:** Concurrent work using 3D point clouds.
*   **DepthAnything-V2 [72]:** Raw mono-depth baseline.

    All baselines are run using open-source implementations on a single Nvidia A100 GPU for fair comparison.

# 6. Results & Analysis

## 6.1. Core Results Analysis
MegaSaM demonstrates significant improvements in both camera tracking and depth estimation.
*   **Camera Tracking:** On the Sintel dataset (Table 1), MegaSaM achieves an ATE of **0.018** (Calibrated) and **0.023** (Uncalibrated), significantly lower than CasualSAM (0.041/0.036) and MonST3R (0.078). This indicates much higher trajectory accuracy.
*   **Depth Estimation:** On Sintel (Table 4), MegaSaM achieves an abs-rel of **0.21** and $\delta _ { 1 . 2 5 }$ of **73.1**, outperforming DepthCrafter (0.27/68.2) and CasualSAM (0.31/64.2).
*   **Runtime:** MegaSaM runs at **1.0s** per video (Sintel), comparable to MonST3R (1.0s) and much faster than CasualSAM (1.3m) or RoDynRF (15m).

**Visual Results:**
*   **Trajectories:** Figure 5 shows MegaSaM's estimated trajectory (red dash) aligns closely with ground truth (blue solid), whereas baselines deviate significantly due to scene dynamics.
*   **Depth:** Figure 6 shows MegaSaM produces temporally consistent depth maps (x-t slices are smooth) compared to the flickering or inconsistent results of baselines.

    ![Figure 5. Visualization of estimated camera trajectories. Due to scene dynamics, our camera estimate (red dash) deviates less from the ground truth camera trajectory (blue solid line) than all other baselines.](images/5.jpg)
    *Figure 5 from the original paper visualizes estimated camera trajectories. The red dashed line (MegaSaM) stays close to the blue solid line (Ground Truth), while other baselines drift significantly in dynamic scenes.*

    ![Figure 6. Visual comparisons of video depths. We compare video depth estimates from our approach and from CasualSAM \[78\] and MonST3R \[76\] by visualizing their depth maps (odd columns) and corresponding $x { - } t$ slices (even columns).](images/6.jpg)
    *Figure 6 from the original paper compares video depths. MegaSaM ("Ours") shows smoother x-t slices (even columns), indicating better temporal consistency compared to CasualSAM and MonST3R.*

## 6.2. Data Presentation (Tables)
The following are the results from **Table 1** of the original paper (Camera estimation on Sintel):

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Calibrated</th>
<th colspan="4">Uncalibrated</th>
</tr>
<tr>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>Time</th>
</tr>
</thead>
<tbody>
<tr>
<td>CasualSAM [78]</td>
<td>0.041</td>
<td>0.023</td>
<td>0.17</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>1.3m</td>
</tr>
<tr>
<td>LEAP-VO [6]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.036</td>
<td>0.013</td>
<td>0.20</td>
<td>-</td>
</tr>
<tr>
<td>ACE-Zero [3]</td>
<td>0.053</td>
<td>0.028</td>
<td>1.26</td>
<td>0.067</td>
<td>0.019</td>
<td>0.47</td>
<td>1.6m</td>
</tr>
<tr>
<td>Particle-SfM [79]</td>
<td>0.062</td>
<td>0.032</td>
<td>1.92</td>
<td>0.057</td>
<td>0.038</td>
<td>1.64</td>
<td>10s</td>
</tr>
<tr>
<td>RoDynRF [34]</td>
<td>0.110</td>
<td>0.049</td>
<td>1.68</td>
<td>0.109</td>
<td>0.051</td>
<td>1.32</td>
<td>15m</td>
</tr>
<tr>
<td>MonST3R [76]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.078</td>
<td>0.038</td>
<td>0.49</td>
<td>1.0s</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.018</b></td>
<td><b>0.008</b></td>
<td><b>0.04</b></td>
<td><b>0.023</b></td>
<td><b>0.008</b></td>
<td><b>0.06</b></td>
<td><b>1.0s</b></td>
</tr>
</tbody>
</table>

The following are the results from **Table 2** of the original paper (Camera estimation on DyCheck):

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Calibrated</th>
<th colspan="4">Uncalibrated</th>
</tr>
<tr>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>Time</th>
</tr>
</thead>
<tbody>
<tr>
<td>CasualSAM [78]</td>
<td>0.185</td>
<td>0.022</td>
<td>0.167</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>2.8m</td>
</tr>
<tr>
<td>LEAP-VO [6]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.209</td>
<td>0.027</td>
<td>0.28</td>
<td>-</td>
</tr>
<tr>
<td>ACE-Zero [3]</td>
<td>0.062</td>
<td>0.012</td>
<td>0.11</td>
<td>0.056</td>
<td>0.012</td>
<td>0.12</td>
<td>0.8s</td>
</tr>
<tr>
<td>Particle-SfM [79]</td>
<td>0.081</td>
<td>0.014</td>
<td>0.20</td>
<td>0.087</td>
<td>0.015</td>
<td>0.29</td>
<td>1.6s</td>
</tr>
<tr>
<td>RoDynRF [34]</td>
<td>0.548</td>
<td>0.074</td>
<td>0.70</td>
<td>0.562</td>
<td>0.087</td>
<td>0.90</td>
<td>6.6m</td>
</tr>
<tr>
<td>MonST3R [76]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.690</td>
<td>0.078</td>
<td>0.54</td>
<td>1.0s</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.020</b></td>
<td><b>0.005</b></td>
<td><b>0.05</b></td>
<td><b>0.020</b></td>
<td><b>0.005</b></td>
<td><b>0.06</b></td>
<td><b>1.0s</b></td>
</tr>
</tbody>
</table>

The following are the results from **Table 4** of the original paper (Video depths):

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Sintel [4]</th>
<th colspan="3">Dycheck [12]</th>
</tr>
<tr>
<th>abs-rel</th>
<th>log-rmse</th>
<th>δ1.25</th>
<th>abs-rel</th>
<th>log-rmse</th>
<th>δ1.25</th>
</tr>
</thead>
<tbody>
<tr>
<td>DA-v2 [72]</td>
<td>0.37</td>
<td>0.55</td>
<td>58.6</td>
<td>0.20</td>
<td>0.27</td>
<td>84.7</td>
</tr>
<tr>
<td>DepthCrafter [20]</td>
<td>0.27</td>
<td>0.50</td>
<td>68.2</td>
<td>0.22</td>
<td>0.29</td>
<td>83.7</td>
</tr>
<tr>
<td>CasualSAM [78]</td>
<td>0.31</td>
<td>0.49</td>
<td>64.2</td>
<td>0.21</td>
<td>0.30</td>
<td>78.4</td>
</tr>
<tr>
<td>MonST3R [76]</td>
<td>0.31</td>
<td>0.43</td>
<td>62.5</td>
<td>0.26</td>
<td>0.35</td>
<td>66.5</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.21</b></td>
<td><b>0.39</b></td>
<td><b>73.1</b></td>
<td><b>0.11</b></td>
<td><b>0.20</b></td>
<td><b>94.1</b></td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The authors conducted ablation studies to validate design choices (Table 5).

**Camera Tracking Ablation:**
*   **w/o mono-init.:** Removing mono-depth initialization increases ATE from 0.019 to 0.038. This confirms initialization is critical for limited baseline videos.
*   **w/o $\mathbf { m } _ { i }$:** Removing object movement map prediction increases RTE significantly (0.008 to 0.127), showing motion masking is vital for dynamic scenes.
*   **w/o u-BA:** Removing uncertainty-aware BA (always using regularization) slightly degrades performance (RRE 0.04 to 0.11), indicating adaptive regularization is better.

**Depth Optimization Ablation:**
*   **w/ ft-pose:** Jointly refining poses during depth optimization worsens pose accuracy (ATE 0.019 to 0.041). Fixing poses is preferred.
*   **w/o new $\mathcal { C } _ { p r i o r }$:** Using the original CasualSAM prior loss reduces depth accuracy ($\delta _ { 1 . 2 5 }$ 73.1 to 72.5).

    ![Figure 2. Ablation on our design choices. From left to right, we visualize cameras and reconstruction from our system (a) without mono-depth initialization, (b) without uncertainty-aware BA, (c) with full configuration. For these difficult near-rotational sequences, our full method produces much better camera and scene geometry.](images/2.jpg)
    *Figure 2 from the original paper shows the ablation on design choices. (a) Without mono-depth initialization, reconstruction fails. (b) Without uncertainty-aware BA, geometry is distorted in rotational sequences. (c) Full configuration produces accurate results.*

The following are the results from **Table 5** of the original paper (Ablation study on Sintel):

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Poses</th>
<th colspan="2">Depth</th>
</tr>
<tr>
<th>ATE</th>
<th>RTE</th>
<th>RRE</th>
<th>Abs-Rel</th>
<th>δ1.25</th>
</tr>
</thead>
<tbody>
<tr>
<td>Droid-SLAM [59]</td>
<td>0.030</td>
<td>0.022</td>
<td>0.50</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o mono-init.</td>
<td>0.038</td>
<td>0.026</td>
<td>0.49</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o mi</td>
<td>0.032</td>
<td>0.127</td>
<td>0.14</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o 2-stage train.</td>
<td>0.035</td>
<td>0.136</td>
<td>0.17</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/o u-BA</td>
<td>0.033</td>
<td>0.013</td>
<td>0.11</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>w/ ft-pose</td>
<td>0.041</td>
<td>0.018</td>
<td>0.33</td>
<td>0.23</td>
<td>71.2</td>
</tr>
<tr>
<td>w/o new Cprior</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.36</td>
<td>72.5</td>
</tr>
<tr>
<td><b>Full</b></td>
<td><b>0.019</b></td>
<td><b>0.008</b></td>
<td><b>0.04</b></td>
<td><b>0.21</b></td>
<td><b>73.1</b></td>
</tr>
</tbody>
</table>

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
MegaSaM presents a robust pipeline for structure and motion estimation from casual dynamic videos. By extending deep visual SLAM with monocular priors and uncertainty-aware optimization, it achieves state-of-the-art accuracy in camera pose and depth estimation. Key findings include:
*   Deep visual SLAM frameworks can be adapted for dynamic scenes with proper training modifications.
*   Uncertainty quantification is essential for handling videos with limited parallax.
*   Consistent video depth can be achieved efficiently without per-video network fine-tuning.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations:
*   **Extreme Dynamics:** Camera tracking can fail if moving objects dominate the entire image or if there are no reliable static features to track.
*   **Camera Intrinsics:** The system cannot handle videos with **varying focal lengths** or strong radial distortion within a single video.
*   **Failure Cases:** Specific failure modes include colinear camera and object motion, where disambiguation is impossible.

    ![该图像是一个包含多个动态场景的视频帧示意图。每个视频帧展示了不同的动态摄像头视角，包括越野摩托车与城市街道的日常生活，最后一组展示了通过新方法生成的场景结构与深度图，体现了系统在复杂动态场景中的应用效果。](images/8.jpg)
    *Figure 8 from the original paper (Appendix) shows failure cases. The first row shows tracking failure when moving objects dominate. The second row shows struggles when camera and object motion are colinear.*

Future work suggested includes incorporating better priors from current vision foundation models to handle these extreme cases.

## 7.3. Personal Insights & Critique
**Strengths:**
*   **Practicality:** The avoidance of test-time fine-tuning makes MegaSaM highly practical for real-world applications compared to methods like CasualSAM.
*   **Robustness:** The uncertainty-aware mechanism is a sophisticated solution to the "degenerate motion" problem (e.g., pure rotation) that plagues SfM.
*   **Integration:** The seamless integration of learning-based priors (DepthAnything) with geometric optimization (BA) represents a strong trend in modern computer vision.

**Critique:**
*   **Dependency on Priors:** The system relies heavily on off-the-shelf mono-depth models (DepthAnything, UniDepth). If these priors fail (e.g., on unusual textures), the system's performance might degrade, though the uncertainty mechanism mitigates this.
*   **Static Assumption in BA:** While motion maps downweight dynamic pixels, the underlying BA formulation still assumes a mostly static background. Extremely dynamic scenes (e.g., a crowd) might still pose challenges.
*   **Compute:** While faster than some baselines, 1.0s per video (likely per short clip) on an A100 is not yet "real-time" for long videos, though it is efficient for offline processing.

**Transferability:**
The **uncertainty-aware regularization** technique could be applied to other optimization problems where data observability varies (e.g., neural rendering, physics simulation). The **two-stage training** (static pretraining + dynamic finetuning) is a robust strategy for adapting models trained on clean data to noisy real-world conditions.