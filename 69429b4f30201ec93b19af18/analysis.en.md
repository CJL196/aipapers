# 1. Bibliographic Information

## 1.1. Title
RELIC: Interactive Video World Model with Long-Horizon Memory

The title clearly states the paper's central topic: the creation of a **world model** capable of generating **interactive video**. The key features highlighted are **long-horizon memory** and, implicitly through "interactive," real-time performance and user control.

## 1.2. Authors
Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan.

The author list comprises a large team, which is common for large-scale deep learning projects. Many of the authors are affiliated with major research labs in computer graphics and AI, suggesting a well-resourced and high-impact project. The notation "First Authors in Random Order" and "Project Lead" indicates a collaborative effort with shared leadership.

## 1.3. Journal/Conference
The paper is available on arXiv, a preprint server. The publication date is listed as December 3rd, 2025, which is likely a placeholder. As a preprint, it has not yet undergone formal peer review for a conference or journal. However, arXiv is a standard platform for disseminating cutting-edge research in fields like machine learning, often before or in parallel with conference submissions.

## 1.4. Publication Year
The paper was submitted to arXiv with a "Date: December 1st, 2025" listed in the text, though this is in the future. The link provided (`/abs/2512.04040`) suggests a submission date in December 2025. Given the current date, this is likely a placeholder or a typo in the paper itself. The content reflects the state-of-the-art as of late 2024 / early 2025.

## 1.5. Abstract
The abstract introduces the challenge of building a true interactive world model, which requires three simultaneous capabilities: **real-time long-horizon streaming**, **consistent spatial memory**, and **precise user control**. The authors argue that existing methods typically fail to achieve all three together, often because long-term memory mechanisms compromise real-time performance.

The paper presents **RELIC**, a unified framework designed to solve this tripartite challenge. RELIC can take a single image and a text prompt to enable long-duration, memory-aware exploration of scenes in real time.

The core methodology involves several key innovations:
1.  **Memory Representation**: It uses highly compressed historical latent tokens stored in the `KV cache` of an autoregressive model. These tokens are encoded with both relative actions and absolute camera poses, enabling implicit 3D-consistent content retrieval with low computational cost.
2.  **Long-Horizon Training**: A bidirectional "teacher" model is fine-tuned to generate long (20-second) videos.
3.  **Efficient Distillation**: A new `self-forcing` paradigm with `replayed back-propagation` is used to distill the slow teacher into a fast, causal "student" generator, making it possible to train on the full context of long video rollouts.

    The resulting 14B-parameter model, trained on a custom Unreal Engine dataset, achieves real-time generation at 16 FPS. It demonstrates superior performance in action following, long-horizon stability, and spatial memory compared to prior work, positioning it as a foundational step for future interactive world models.

## 1.6. Original Source Link
*   **Original Source Link:** `https://arxiv.org/abs/2512.04040`
*   **PDF Link:** `https://arxiv.org/pdf/2512.04040v1.pdf`

    The paper is available as a preprint on arXiv, meaning it is a research article that has not yet been peer-reviewed for publication in a journal or conference.

# 2. Executive Summary

## 2.1. Background & Motivation
The ultimate goal of a "world model" is to create a learnable, internal simulation of an environment. A truly useful world model for applications like virtual reality, gaming, or robotics must be **interactive**. This interactivity imposes three demanding and often conflicting requirements:

1.  **Real-Time Performance**: The model must generate video frames faster than a human can perceive the delay (e.g., >15 FPS), allowing for fluid user interaction.
2.  **Precise User Control**: The model must faithfully translate user inputs (like keyboard strokes or mouse movements) into corresponding camera motions in the generated video.
3.  **Long-Horizon Spatial Memory**: As a user explores a scene, the model must remember what it has "seen." If the user turns back to a previously visited location, the scene should appear consistent with how it looked before.

    The core problem is that existing approaches typically sacrifice one of these pillars to achieve the others. For example, methods that use large memory buffers or complex 3D representations to ensure long-term consistency are often too slow for real-time interaction. The memory and computational costs of processing long sequences of past information create a bottleneck that conflicts directly with the need for low-latency generation.

RELIC's innovative entry point is to design a unified framework that tackles all three challenges simultaneously. The key idea is to create a highly efficient memory mechanism that is powerful enough for long-term consistency but lightweight enough for real-time performance.

## 2.2. Main Contributions / Findings
The paper makes several key contributions to the field of interactive video generation:

1.  **A Unified Framework (RELIC)**: The primary contribution is the RELIC model itself, which is the first to successfully integrate real-time streaming, precise control, and long-horizon spatial memory for interactive world exploration from a single starting image.

2.  **Camera-Aware Compressed Memory**: RELIC introduces a novel memory mechanism. Instead of storing full, high-fidelity past information, it stores highly compressed latent tokens in the model's `KV cache`. Crucially, these tokens are enriched with absolute camera pose information, allowing the model to implicitly retrieve relevant scene content based on the current viewpoint, thus ensuring 3D consistency.

3.  **Long-Context Distillation with Replayed Back-propagation**: To train a model capable of generating long, consistent videos, the authors first fine-tune a powerful teacher model to generate 20-second sequences. They then introduce a memory-efficient distillation technique called `replayed back-propagation`. This allows a fast student model to learn from the full 20-second rollouts of the teacher without the prohibitive memory costs of standard backpropagation through time.

4.  **A High-Quality Curated Dataset**: The authors created a new dataset rendered in Unreal Engine (UE). This dataset features diverse scenes, precise 6-DoF camera trajectory data, and is carefully filtered to ensure high-quality, physically plausible motions, which is crucial for training a controllable model.

    The main finding is that this combination of architectural and training innovations works. **RELIC achieves real-time (16 FPS) generation of videos up to 20 seconds long, exhibiting significantly more accurate action following and more robust spatial memory (e.g., remembering objects when returning to a location) than previous state-of-the-art models like `Matrix-Game-2.0` and `Hunyuan-GameCraft`.**

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand RELIC, several core concepts from deep learning are essential:

*   **World Models**: A world model is a generative neural network that learns a compressed spatial and temporal representation of an environment. It can be trained to predict future states of the environment based on current states and actions. In this paper, the "world" is a static 3D scene, and the model learns to generate new video frames as a user "moves" a virtual camera through it.

*   **Video Diffusion Models**: These are a class of generative models that have achieved state-of-the-art results in image and video synthesis. They work in two stages:
    1.  **Forward Process**: A clean video is gradually destroyed by adding a small amount of Gaussian noise over many steps.
    2.  **Reverse Process**: A neural network (often a U-Net or a Transformer) is trained to reverse this process. It learns to take a noisy video and a timestep as input and predict the noise that was added. By iteratively applying this denoising network starting from pure noise, a new, clean video can be generated.

*   **Autoregressive (AR) Models**: An AR model generates a sequence one element at a time, where each new element is conditioned on the previously generated elements. For video, this means generating the next frame (or a small chunk of frames) based on the frames that came before it. This is inherently sequential and well-suited for streaming applications.

*   **Model Distillation**: This is a technique to transfer knowledge from a large, slow, but powerful "teacher" model to a smaller, faster "student" model. The student is trained to mimic the teacher's output distribution. In RELIC's context, the teacher is a slow, multi-step diffusion model that produces high-quality long videos, and the student is a fast, few-step AR model optimized for real-time inference.

*   **KV Cache in Transformers**: The Transformer architecture, which is central to models like RELIC, relies on the `self-attention` mechanism. During autoregressive generation, the `Key (K)` and `Value (V)` vectors are computed for each input token. To avoid recomputing these for past tokens at every new step, they are stored in a `KV cache`. For generating the next token, the model only computes the `Query (Q)` for the new token and attends to all the $K$ and $V$ vectors in the cache. While this speeds up inference, the cache size grows linearly with the sequence length, becoming a memory bottleneck for long videos.

*   **6-DoF (Degrees of Freedom)**: In 3D space, any rigid body has six degrees of freedom. These describe its movement and orientation. They are typically broken down into:
    *   **3 Translational DoF**: Movement along the X, Y, and Z axes (e.g., forward/backward, left/right, up/down).
    *   **3 Rotational DoF**: Rotation around the X, Y, and Z axes, known as **roll**, **pitch**, and **yaw**.
        RELIC's precise user control is based on mapping user inputs to changes in these 6-DoF camera poses.

## 3.2. Previous Works
The authors build upon a rich history of research in video generation and world modeling.

*   **Autoregressive Video Diffusion**: Models like `StreamDiffusion` and `Rolling-Diffusion` pioneered real-time video generation by combining diffusion models with an AR framework. They typically use a sliding window over the `KV cache` to manage memory, but this limits their ability to recall information from the distant past, leading to a lack of long-horizon consistency. RELIC's compressed memory cache is a direct improvement over this simple sliding-window approach.

*   **Diffusion Model Distillation**: The idea of making diffusion models fast enough for real-time use was popularized by **Consistency Models** and later refined by frameworks like **Diffusion-Forcing** and **Score-Matching Distillation (DMD)**. DMD, used in RELIC, trains the student to match the "score" (gradient of the log probability density) of the teacher's output distribution. The key formula for the DMD loss gradient, which RELIC builds upon, is:
    \$
    \nabla_{\theta} \mathcal{L}_{KL} \approx - \mathbb{E}_{u} \left[ \int (s^{\mathrm{data}}(\Psi(G_{\theta}(\epsilon, c_{\mathrm{text}}), u) - s^{\mathrm{gen}}(\Psi(G_{\theta}(\epsilon, c_{\mathrm{text}}), u))) \frac{dG_{\theta}(\epsilon, c_{\mathrm{text}})}{d\theta} d\epsilon \right]
    \$
    Here, $G_{\theta}$ is the student generator, $s^{\mathrm{data}}$ and $s^{\mathrm{gen}}$ are the score functions from the teacher (approximating the data distribution) and a model of the student's own output, respectively. RELIC's innovation is developing a method to compute this over very long sequences.

*   **Long Video Generation**: Models like `Long-Vid-Gen` and `Long-Live-Video` have focused on generating minute-scale videos. They achieve this by training on longer video clips. RELIC adopts this principle by fine-tuning its teacher model on 20-second clips, providing the necessary long-context supervision for the student.

*   **Interactive World Models**: Recent models like Google's `Genie`, Wayve's `GAIA-1`, and Tencent's `Matrix-Game` and `Hunyuan-GameCraft` have demonstrated the ability to generate interactive experiences from user actions. However, the paper points out their limitations: `Genie` and `Matrix-Game` often lack strong spatial memory, while others are not optimized for real-time streaming. RELIC aims to combine the strengths of these models while addressing their weaknesses.

## 3.3. Technological Evolution
The field has progressed from generating short, non-interactive video clips to creating dynamic, controllable virtual worlds.
1.  **Early Video Generation**: Focused on short, unconditional video synthesis.
2.  **Controllable Generation**: Introduction of text prompts and other conditions to control video content.
3.  **Real-Time Streaming**: Development of AR diffusion and distillation techniques to enable low-latency generation.
4.  **Interactive World Models**: Application of these techniques to create game-like environments that respond to user actions.
5.  **Long-Horizon Memory (RELIC's focus)**: The current frontier, aiming to make these interactive worlds spatially and temporally consistent over long periods of exploration. RELIC sits at this cutting edge.

## 3.4. Differentiation Analysis
Compared to its predecessors, RELIC's primary innovation is its **holistic approach**.

*   **vs. `StreamDiffusion` / `Rolling-Diffusion`**: These models achieve real-time speed but have poor long-term memory due to their simple sliding-window KV caches. **RELIC introduces a compressed, camera-aware memory cache that retains information from the entire history**, not just the recent past.

*   **vs. `Genie` / `Matrix-Game 2.0`**: These models are interactive but often fail to maintain spatial consistency. For example, they might "forget" an object if the camera looks away and then back again. **RELIC's camera-pose-encoded memory is explicitly designed to solve this problem**, allowing it to retrieve content based on viewpoint and maintain a coherent world.

*   **vs. Methods with Explicit 3D Representations**: Some models try to maintain consistency by building an explicit 3D representation of the scene (like a point cloud or mesh). This can be computationally expensive and may not scale well. **RELIC uses an implicit representation stored in the latent space**, which is more efficient and integrates seamlessly into the Transformer architecture.

*   **vs. Previous Distillation Methods**: Standard distillation techniques struggle with the massive memory requirements of backpropagating through long video sequences. **RELIC's `replayed back-propagation` is a novel, memory-efficient algorithm that makes it feasible to train a student on a long-horizon (20-second) teacher distribution.**

# 4. Methodology

## 4.1. Principles
The core principle of RELIC is to build a fast, autoregressive **student model** that can generate interactive video in real time by learning from a powerful but slow **teacher model**. The entire framework is designed to overcome the fundamental tension between **long-horizon memory** and **real-time performance**. This is achieved through three main pillars:
1.  **A high-quality, long-duration teacher**: A model that can generate spatially and temporally consistent 20-second videos, serving as the "gold standard" for the student to learn from.
2.  **An efficient memory architecture**: A novel KV cache mechanism that compresses distant past information, enriched with camera pose data, allowing the student to "remember" the entire scene with minimal computational overhead.
3.  **A scalable distillation paradigm**: A training method that allows the student to learn from the teacher's full, long-duration rollouts without running out of GPU memory.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Data Curation and Annotation
The foundation of RELIC is a high-quality dataset. The authors construct this dataset using **Unreal Engine (UE)**.

*   **Data Collection**: Human operators navigate 350 diverse, photorealistic static UE scenes. During navigation, full 6-DoF camera trajectories (position and orientation) are recorded and rendered into 720p videos. This results in over 1600 minutes of training data. The distribution of video durations is shown in Figure 2.

    ![Figure 2 Dataset statistics visualization. Left:video duration distribution; Right: action distribution.](images/2.jpg)
    *该图像是图表，展示了我们的 curated UE 数据集中的视频时长分布和动作分布。左侧为视频时长的概率密度图，其中包含均值（75秒）、中位数（63秒）和最大值（545秒）；右侧显示了不同动作的百分比，最大值为30.6%。*

*   **Filtering**: Raw videos are rigorously filtered to remove artifacts that could teach the model bad habits, such as unnatural camera motion (e.g., abrupt rotations), viewpoint jitters, poor lighting, and rendering glitches.

*   **Action Annotation**: The model needs to learn a mapping from discrete user actions to video changes. The authors convert the continuous 6-DoF camera trajectories into per-frame action labels. This is detailed in Algorithm 1.

    For each frame $t$, they compute the **relative camera motion** between frame $t$ and $t+1$.

    1.  **Relative Translation**: The change in position is calculated in the camera's own coordinate system.
        \$
        \Delta \mathbf{P}_{t}^{c} = R_{t} (\mathbf{P}_{t+1} - \mathbf{P}_{t})
        \$
        *   $\mathbf{P}_{t}$ and $\mathbf{P}_{t+1}$ are the absolute camera positions in world coordinates at times $t$ and $t+1$.
        *   $R_{t}$ is the world-to-camera rotation matrix at time $t$. It transforms the world-space displacement vector $(\mathbf{P}_{t+1} - \mathbf{P}_{t})$ into the camera's local coordinate system, giving the egocentric motion $\Delta \mathbf{P}_{t}^{c}$.
        *   The components of this vector correspond to forward/backward, left/right, and up/down motion.

    2.  **Relative Rotation**: The change in orientation is calculated.
        \$
        \Delta R_{t}^{c} = R_{t+1} (R_{t})^{T}
        \$
        *   This formula gives the rotation that transforms the camera's orientation from time $t$ to $t+1$. $(R_t)^T$ is the camera-to-world rotation.
        *   This relative rotation matrix $\Delta R_t^c$ is then decomposed into **yaw, pitch, and roll** angles, which correspond to looking left/right, up/down, and rolling, respectively.

            These relative motions are then discretized and binned to form the 13-dimensional action vector $\mathcal{A}_t$ used for training.

*   **Data Augmentation**: To ensure the model frequently encounters scenarios where it must revisit past locations, a "palindrome-style" data augmentation is used. A video segment is concatenated with its time-reversed version, forcing the model to generate a path that goes out and then comes back, testing its long-term memory. This is illustrated in Figure 3.

    ![该图像是一个示意图，展示了RELIC框架在3D场景中的渲染与处理流程。图中展示了摄像机轨迹、动作映射以及时间反转增强等要素，旨在实现长久的交互式视频世界建模。](images/3.jpg)
    *该图像是一个示意图，展示了RELIC框架在3D场景中的渲染与处理流程。图中展示了摄像机轨迹、动作映射以及时间反转增强等要素，旨在实现长久的交互式视频世界建模。*

### 4.2.2. The RELIC Model Architecture
RELIC consists of a teacher-student pair, both based on the `W.A.L.T-2.1` architecture, which uses a Spatio-Temporal VAE (`ST-VAE`) to encode video into latents and a Diffusion Transformer (`DiT`) to operate on them.

**1. The Long-Horizon Teacher Model**

*   **Base Architecture**: A 14B-parameter bidirectional video diffusion model.
*   **Action Conditioning**: To make the model controllable, it's conditioned on two types of signals:
    *   **Relative Actions ($\mathcal{A}_t$)**: The 13-dimensional per-frame action vector. These are embedded and added to the video latent tokens, directly influencing the immediate motion.
    *   **Absolute Camera Poses ($\mathbf{P}_t, R_t$)**: The absolute position and orientation of the camera in the world. These are obtained by integrating the relative actions over time:
        \$
        \mathbf{P}_{t} = \sum_{i=1}^{t} (R_{i})^{T} \Delta \mathbf{P}_{i}^{c}, \quad R_{t} = \prod_{i=1}^{t} \Delta R_{i}^{c}
        \$
        These pose embeddings are added to the **Query (Q) and Key (K) projections** in the self-attention layers. This is a crucial design choice: it makes the attention mechanism viewpoint-aware. When the model needs to generate content for a specific location, the absolute pose information helps it "look up" relevant information from past frames that were generated at or near the same location.
*   **Long-Horizon Training**: The original `W.A.L.T` was trained on 5-second videos. The authors fine-tune this teacher model on their 20-second UE dataset. To handle the longer sequence length, they extend the context window of the model's Positional Embeddings (`RoPE`). This allows the teacher to generate high-quality, consistent 20-second videos that will serve as supervision for the student.

    The overall teacher architecture is shown in Figure 4.

    ![该图像是示意图，展示了RELIC框架的工作流程。输入图像经过噪声处理后，通过DiT模块生成20秒的视频，涉及视频潜在特征的提取与处理，包括自注意力机制和跨注意力机制等结构。](images/4.jpg)

    **2. The Autoregressive Student Model and Memory Mechanism**

*   **Architecture**: The student is an autoregressive version of the same `W.A.L.T` model. It generates video frame-by-frame (or chunk-by-chunk) in a single forward pass, making it fast.
*   **The Memory Dilemma**: A standard AR model would need to store the `KV cache` for all previously generated frames. For a 20-second video at 16 FPS (320 frames), this cache would become enormous, making real-time generation impossible.
*   **RELIC's Solution: Compressed Spatial Memory**: RELIC introduces a dual-branch `KV cache` to solve this:
    1.  **Rolling-Window Cache**: Stores the full, uncompressed $K$ and $V$ tokens for a small number of recent frames (e.g., within a window $w$). This ensures high-fidelity generation for the immediate future.
    2.  **Compressed Long-Horizon Cache**: For all frames older than the window $w$, it stores **spatially downsampled** $K$ and $V$ tokens. The paper uses an interleaved compression schedule (e.g., no compression, 2x downsampling, 4x downsampling). This dramatically reduces the number of tokens that need to be stored and attended to.
    
        This strategy reduces the total token count by about 4x, leading to a proportional reduction in memory usage and attention computation (FLOPs), which is the key to enabling real-time performance over long horizons.

### 4.2.3. The Distillation Framework
The student model is trained to mimic the teacher using a novel distillation framework. The process is visualized in Figure 5.

![该图像是一个示意图，展示了RELIC框架中的教师和学生模型之间的关系。图中左侧的 $G_{teacher}$ 处理文本提示、输入图像和动作控制，生成ODE轨迹；右侧的 $G_{student}$ 则通过MSE Loss从压缩内存中检索信息。整体结构体现了模型在长时间序列生成中的设计思路。](images/5.jpg)
*该图像是一个示意图，展示了RELIC框架中的教师和学生模型之间的关系。图中左侧的 $G_{teacher}$ 处理文本提示、输入图像和动作控制，生成ODE轨迹；右侧的 $G_{student}$ 则通过MSE Loss从压缩内存中检索信息。整体结构体现了模型在长时间序列生成中的设计思路。*

*   **ODE Initialization with Hybrid Forcing**: The training starts by initializing the student model to approximate the Ordinary Differential Equation (ODE) trajectories of the teacher. During this phase, `Hybrid Forcing` is used. For a given training sequence, the initial part is fed to the model as clean, ground-truth latents (but spatially compressed, to match the memory format), while the latter part is fed as noisy latents that the model must denoise. This hybrid approach was found to provide a better balance between fast convergence and robust memory retrieval.

*   **Long-Video Distillation with Replayed Back-propagation**: This is the core training innovation, designed to overcome the memory limits of standard backpropagation. The goal is to compute the gradient of the `DMD loss` over a full 20-second student rollout.

    The standard approach of running the full AR generation and then backpropagating would require storing the computation graph for the entire sequence, which is too large for current GPUs. `Replayed back-propagation` circumvents this with a clever block-wise procedure, illustrated in Figure 6.

    ![该图像是示意图，展示了 RELIC 模型中不同组件的结构和交互，包括学生生成器、教师生成器及 KV 缓存的操作过程。图中包含了缓存得分差异和完整序列推理的执行步骤。](images/6.jpg)
    *该图像是示意图，展示了 RELIC 模型中不同组件的结构和交互，包括学生生成器、教师生成器及 KV 缓存的操作过程。图中包含了缓存得分差异和完整序列推理的执行步骤。*

    1.  **Step 1: Forward Rollout (No Grad)**: First, the student model generates the entire 20-second sequence of video latents $\hat{\mathbf{x}}_{0:L}$ autoregressively. This is done with gradient calculation disabled (`stop-grad`), so it consumes very little memory.
    2.  **Step 2: Compute Score Differences**: The generated sequence $\hat{\mathbf{x}}_{0:L}$ is passed through frozen "score models" (derived from the teacher) to compute the score-difference maps $\Delta \hat{s}_{0:L}$. This map represents the "error" or "correction" signal for each token in the generated sequence.
        \$
        \Delta \hat{s}_{0:L} = s^{\mathrm{data}}(\hat{\mathbf{x}}_{0:L}) - s^{\mathrm{gen}}(\hat{\mathbf{x}}_{0:L})
        \$
    3.  **Step 3: Replay and Back-propagate Block-by-Block**: The framework then "replays" the generation process, but this time with gradient calculation enabled, one block (a small chunk of frames) at a time.
        *   For block $l$, it re-runs the student's forward pass for just that block, conditioned on the previously generated (and cached) history.
        *   It then backpropagates the pre-computed score difference $\Delta \hat{s}_l$ for that block through the block's computation graph.
        *   The resulting gradients for the model parameters $\theta$ are **accumulated**.
        *   After backpropagation for block $l$, its computation graph is immediately discarded, freeing up memory.
    4.  **Step 4: Update Parameters**: After replaying all blocks and accumulating gradients from the entire 20-second sequence, a single optimization step is performed to update the model parameters $\theta$.

        This procedure effectively computes the full gradient for the long sequence while only ever holding the computation graph for one small block in memory at a time.

### 4.2.4. Runtime Efficiency Optimizations
To achieve the target 16 FPS, several low-level optimizations are applied:
*   Kernel fusion, `RMSNorm`, and optimized `RoPE` to reduce GPU overhead.
*   Using the `FP8 E4M3` numerical format for storing cached latents to halve memory usage.
*   `FlashAttention v3` to accelerate the attention computation.
*   A hybrid parallelization strategy combining sequence parallelism and tensor parallelism to efficiently distribute the workload across multiple GPUs.

# 5. Experimental Setup

## 5.1. Datasets
*   **Training Dataset**: The primary dataset is the authors' custom-curated collection of **1600 minutes** of video from **350 photorealistic static scenes** rendered in **Unreal Engine**. The key feature is the high-quality, frame-aligned 6-DoF camera pose annotations, which are essential for training the action-following and memory retrieval capabilities.
*   **Test Dataset**: For quantitative and qualitative evaluation, a benchmark test set was constructed using **220 images** sourced from Adobe Stock. These images span both realistic and fantasy scenes. For each starting image, a predefined action script is used to generate videos from all models, ensuring a fair comparison.

## 5.2. Evaluation Metrics
The paper evaluates RELIC on visual quality and action accuracy.

*   **Visual Quality (User Study)**: Human evaluators were asked to rate videos on a scale for three aspects: `Image Quality`, `Aesthetic` quality, and an overall `Average Score`. This captures the subjective perceptual quality of the generated videos.

*   **Action Accuracy (Relative Pose Error, RPE)**: This metric quantifies how accurately the model follows the commanded actions.
    1.  **Conceptual Definition**: RPE measures the drift between the camera trajectory generated by the model and the ground-truth trajectory that would result from perfectly executing the input action sequence. Lower RPE means the generated camera motion is closer to the intended motion.
    2.  **Mathematical Formula**: First, the generated trajectory $Q$ and ground-truth trajectory $P$ are aligned using a similarity transformation $S$ (Sim(3) alignment) to remove global differences in scale, rotation, and translation. The RPE over a time interval $\Delta t$ is then calculated as the error in the relative transformation over that interval:
        \$
        \text{RPE}_i = \left( (P_i^{-1} P_{i+\Delta t})^{-1} (Q_i^{-1} Q_{i+\Delta t}) \right)
        \$
        The final reported RPE is usually the Root Mean Squared Error (RMSE) of the translational and rotational parts of this error term over all time steps.
    3.  **Symbol Explanation**:
        *   $P_i, Q_i$: The absolute camera poses (as transformation matrices) at time step $i$ for the ground-truth and generated trajectories, respectively.
        *   $P_i^{-1} P_{i+\Delta t}$: The relative pose transformation from time $i$ to $i+\Delta t$ in the ground-truth trajectory.
        *   $Q_i^{-1} Q_{i+\Delta t}$: The relative pose transformation from time $i$ to $i+\Delta t$ in the generated trajectory.

## 5.3. Baselines
RELIC is compared against two contemporary state-of-the-art interactive world/game models:
*   **`Matrix-Game-2.0`**: A foundation model for interactive worlds, known for its ability to generate gameplay footage.
*   **`Hunyuan-GameCraft`**: Another large-scale model designed for generating interactive game-like videos from action inputs.

    These are strong baselines as they represent the leading edge of research in the same problem domain.

# 6. Results & Analysis

The paper presents a comprehensive set of results, including a comparison table of existing models, quantitative metrics, and qualitative examples.

The following is the comparison table from the original paper (Table 1):

<table><tr><td></td><td>The Matrix</td><td>Genie-2</td><td>GameCraft</td><td>Yume</td><td>Yan</td><td>Matrix-Game 2.0</td><td>Genie-3</td><td>RELIC (ours)</td></tr><tr><td>Data Source</td><td>AAA Games</td><td>Unknown</td><td>AAA Games</td><td>Sekai</td><td>3D game</td><td>Minecraft+UE+Sekai</td><td>Unknown</td><td>UE</td></tr><tr><td>Action Space</td><td>4T4R</td><td>5T4R2E</td><td>4T4R</td><td>4T4R</td><td>7T2R</td><td>4T</td><td>5T4R1E</td><td>6T6R</td></tr><tr><td>Resolution</td><td>720×1280</td><td>720×1280</td><td>720×1280</td><td>544×960</td><td>1080×1920</td><td>352×640</td><td>704×1280</td><td>480×832</td></tr><tr><td>Speed</td><td>8-16 FPS</td><td>Unknown</td><td>24 FPS</td><td>16 FPS</td><td>60 FPS</td><td>25 FPS</td><td>24 FPS</td><td>16 FPS</td></tr><tr><td>Duration</td><td>Infinite</td><td>10-20 sec</td><td>1 min</td><td>20 sec</td><td>Infinite</td><td>1 min</td><td>1 min</td><td>20 sec</td></tr><tr><td>Generalization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Memory</td><td>None</td><td></td><td></td><td>None</td><td>None</td><td>None</td><td></td><td></td></tr><tr><td>Model Size</td><td>2.7B</td><td>Unknown</td><td>13B</td><td>14B</td><td>Unknown</td><td>1.3B</td><td>Unknown</td><td>14B</td></tr></table>

This table positions RELIC among other leading models, highlighting its unique combination of a large 6T6R (6 translational, 6 rotational) action space, real-time speed (16 FPS), and explicit focus on memory.

## 6.1. Core Results Analysis
The main quantitative results are presented in Table 2, comparing RELIC to the baselines on the Adobe Stock test set.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">Visual quality ↑</th>
<th colspan="2">Action accuracy (RPE ↓)</th>
</tr>
<tr>
<th>Average Score†</th>
<th>Image Quality</th>
<th>Aesthetic</th>
<th>Trans</th>
<th>Rot</th>
</tr>
</thead>
<tbody>
<tr>
<td>Matrix-Game-2.0 (He et al., 2025)</td>
<td>0.7447</td>
<td>0.6551</td>
<td>0.4931</td>
<td>0.1122</td>
<td>1.48</td>
</tr>
<tr>
<td>Hunyuan-GameCraft (Li et al., 2025a)</td>
<td>0.7885</td>
<td>0.6737</td>
<td>0.5874</td>
<td>0.1149</td>
<td>1.23</td>
</tr>
<tr>
<td>RELIC (ours)</td>
<td><strong>0.8015</strong></td>
<td>0.6665</td>
<td><strong>0.5967</strong></td>
<td><strong>0.0906</strong></td>
<td><strong>1.00</strong></td>
</tr>
</tbody>
</table>

*   **Visual Quality**: RELIC achieves the highest `Average Score` and `Aesthetic` score, indicating that users found its outputs to be the most visually appealing overall. Its `Image Quality` is competitive with `Hunyuan-GameCraft`. This shows that the distillation process and memory compression do not degrade the visual fidelity of the generated video.

*   **Action Accuracy**: This is where RELIC shows a clear advantage. It achieves the lowest (best) `Relative Pose Error (RPE)` for both translation (`Trans`) and rotation (`Rot`). This result strongly validates the effectiveness of RELIC's action conditioning mechanism. The model is significantly better at translating user commands into the intended camera motion compared to the baselines.

## 6.2. Qualitative Analysis
The qualitative examples provide intuitive evidence for RELIC's capabilities.

*   **Long-Horizon Memory (Figure 1 and 9)**: The most compelling showcase is the "revisiting" scenario. In Figure 9, the camera moves away from a bench and then rotates back. Both `Hunyuan-GameCraft` and `Matrix-Game-2.0` "forget" the bench and generate inconsistent scenery. In contrast, RELIC successfully regenerates the bench, demonstrating robust spatial memory retrieval.

    ![该图像是实验结果展示图，呈现了生成的场景在不同时间点（0s、1s、9s、12s、20s）下的变化，并比较了我们的模型、Hunyuan和Matrix 2.0的表现。图中标号对应了旋转动作的指示，说明了不同时间点的视图变化。](images/9.jpg)
    *该图像是实验结果展示图，呈现了生成的场景在不同时间点（0s、1s、9s、12s、20s）下的变化，并比较了我们的模型、Hunyuan和Matrix 2.0的表现。图中标号对应了旋转动作的指示，说明了不同时间点的视图变化。*

*   **Action Accuracy (Figure 8)**: This figure visually demonstrates the quantitative RPE results. When commanded to `Tilt Up`, RELIC correctly pans the camera upward, while `Matrix-Game-2.0` drifts sideways. When commanded to `Truck Left` (a sideways translation), RELIC executes the motion correctly, while `Hunyuan-GameCraft` performs a rotation (`Pan Left`) instead. This highlights RELIC's superior control fidelity.

    ![该图像是对比图，展示了输入图像与四个不同方法生成的图像效果，包括Matrix-Game 2.0、Hunyuan-GameCraft和Ours。其中，Ours方法在记忆和空间一致性方面表现出色，展现了较强的图像生成能力。](images/8.jpg)
    *该图像是对比图，展示了输入图像与四个不同方法生成的图像效果，包括Matrix-Game 2.0、Hunyuan-GameCraft和Ours。其中，Ours方法在记忆和空间一致性方面表现出色，展现了较强的图像生成能力。*

*   **Generalization and Control (Figure 7)**: RELIC demonstrates the ability to generate diverse content beyond its training data (e.g., fantasy castles, sci-fi scenes) and allows users to adjust the exploration speed by scaling the action coefficient, showcasing its flexibility.

    ![该图像是多个示意图，展示了RELIC框架的不同功能，包括多样化艺术风格的泛化、可调速度和多键控制等。每个部分的图像展示了不同的场景及移动方式，便于理解其交互式世界建模的能力。](images/7.jpg)
    *该图像是多个示意图，展示了RELIC框架的不同功能，包括多样化艺术风格的泛化、可调速度和多键控制等。每个部分的图像展示了不同的场景及移动方式，便于理解其交互式世界建模的能力。*

*   **Comparison with Commercial System (Figure 10)**: A comparison with `Marble`, a commercial product, shows RELIC generating cleaner results without the "Gaussian floater" artifacts present in the competitor's output, suggesting high generation quality.

    ![该图像是一个示意图，展现了RELIC模型与其他方法在场景生成方面的比较。上部展示了输入图像和Marble（World Labs）方法生成的结果，下部展示了RELIC生成的场景，强调其真实感和细节表现。](images/10.jpg)
    *该图像是一个示意图，展现了RELIC模型与其他方法在场景生成方面的比较。上部展示了输入图像和Marble（World Labs）方法生成的结果，下部展示了RELIC生成的场景，强调其真实感和细节表现。*

## 6.3. Ablation Studies / Parameter Analysis
The paper does not include a dedicated section for ablation studies, which would have been valuable to quantitatively isolate the contribution of each component (e.g., compressed memory vs. sliding window, replayed back-propagation vs. standard distillation on shorter clips, absolute pose conditioning vs. not). However, the strong overall results implicitly validate the effectiveness of the integrated system.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully presents **RELIC**, a novel interactive video world model that makes significant progress toward the ambitious goal of achieving **real-time streaming, precise user control, and long-horizon spatial memory** within a single, unified framework. By integrating a camera-aware compressed memory mechanism with a scalable, long-context distillation paradigm (`replayed back-propagation`), RELIC demonstrates state-of-the-art performance. It excels in both quantitative metrics, particularly action-following accuracy, and qualitative assessments of spatial consistency. The work establishes a strong foundation for the next generation of general-purpose world simulators with potential applications in AI, gaming, and immersive content creation.

## 7.2. Limitations & Future Work
The authors candidly acknowledge several limitations:
1.  **Static Worlds**: The model is trained on static scenes. It cannot simulate dynamic objects, physics, or interactions between entities within the world. The world is a "background" to be explored, not a living environment.
2.  **Limited Scene Scale**: The model struggles with extremely long exploration trajectories, suggesting that the compressed memory, while effective for 20-second clips, may have capacity limits.
3.  **Inference Latency**: The model relies on a few-step denoising process during inference. Moving to a true one-step generation model would further reduce latency and improve real-time responsiveness, especially on resource-constrained hardware.
4.  **Dataset Dependency**: The model's capabilities are heavily influenced by the curated UE dataset. Generalizing to more complex and chaotic real-world video remains a challenge.

## 7.3. Personal Insights & Critique
This paper is a strong piece of engineering and a significant step forward for interactive AI.

*   **Pragmatic Innovation**: The two key innovations, `compressed KV cache` and `replayed back-propagation`, are highly pragmatic. They don't invent a completely new modeling paradigm but instead provide clever and effective solutions to well-known computational bottlenecks. The `replayed back-propagation` technique, in particular, is a general-purpose tool that could be valuable for training any long-sequence autoregressive model.

*   **Implicit vs. Explicit 3D**: The choice to use an *implicit* 3D representation via pose-encoded latents is insightful. It avoids the complexities and computational costs of maintaining an explicit geometric structure (like a NeRF or a mesh) while still achieving impressive 3D consistency. This shows that "good enough" 3D awareness can be baked directly into the attention mechanism of a 2D video model.

*   **Critique on "World Model" Terminology**: While impressive, calling RELIC a "world model" might be a slight overstatement. It is more accurately a "scene exploration model" or an "interactive viewpoint generation model." True world models, as envisioned by pioneers like Jürgen Schmidhuber, are expected to learn the underlying rules and dynamics of their environment (i.e., the "physics"). RELIC's world is static and lacks causal dynamics. This is not a flaw of the paper, which is clear about its scope, but a point of clarification regarding the terminology's evolution in the field.

*   **Future Impact**: RELIC lays a clear path for future work. The next logical steps would be to incorporate object-level understanding, physical dynamics, and multi-agent interaction into this real-time, long-horizon framework. The model's ability to generate consistent, controllable environments could be a game-changer for creating training grounds for embodied AI agents or for rapid prototyping in the creative industries. The efficiency of the memory system makes it a promising architecture to build upon for these more ambitious goals.