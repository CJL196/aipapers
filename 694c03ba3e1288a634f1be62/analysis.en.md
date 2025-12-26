# 1. Bibliographic Information

## 1.1. Title
Generalizable Humanoid Manipulation with 3D Diffusion Policies

## 1.2. Authors
Yanjie Ze (Stanford University), Zixuan Chen (Simon Fraser University), Wenhao Wang (UPenn), Tianyi Chen (UPenn), Xialin He (UIUC), Ying Yuan (CMU), Xue Bin Peng (Simon Fraser University), Jiajun Wu (Stanford University).

## 1.3. Journal/Conference
This paper was published as a research article and made available via arXiv (2024). It involves researchers from top-tier institutions (Stanford, CMU, UPenn) known for their leading roles in robotics and computer vision.

## 1.4. Publication Year
Published on October 14, 2024.

## 1.5. Abstract
The research addresses the challenge of autonomous humanoid manipulation in diverse, unseen environments. Traditionally, humanoid skills have been restricted to the specific scenes where they were trained due to the high cost of data collection and the limited generalization of learning algorithms. The authors present a system integrating:
1.  A **whole-upper-body robotic teleoperation system** to collect human-like data.
2.  A **25-degree-of-freedom (DoF) humanoid robot platform** equipped with a 3D LiDAR sensor.
3.  The **Improved 3D Diffusion Policy (iDP3)**, a learning algorithm designed to handle noisy human data and generalize to new scenes.
    Evaluated with over 2000 real-world trials, the system enables a full-sized humanoid robot to perform tasks like picking, pouring, and wiping in unseen scenarios (kitchens, offices) using data collected from only a single training scene.

## 1.6. Original Source Link
*   **PDF Link:** [https://arxiv.org/pdf/2410.10803v3.pdf](https://arxiv.org/pdf/2410.10803v3.pdf)
*   **Project Website:** [https://humanoid-manipulation.github.io](https://humanoid-manipulation.github.io)

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The central problem is that while humanoid robots have improved in hardware, their **autonomous manipulation skills** are usually "brittle"—they only work in the exact environment where they were trained. 

**Why is this important?**
For robots to be truly useful in homes or hospitals, they must handle "unstructured environments" (places that change, like a messy kitchen).

**Challenges in Prior Research:**
*   **Data Scarcity:** Collecting data in many different "wild" scenes is expensive and time-consuming.
*   **Algorithmic Limitations:** Most current models use 2D images. 2D models often overfit to specific colors or backgrounds. When the robot moves to a new room, the visual changes confuse the model.
*   **Hardware Complexity:** Humanoid robots have many moving parts (degrees of freedom), making control and teleoperation difficult.

**Innovative Idea:**
The authors propose using **3D data** (point clouds) instead of 2D images. By focusing on the 3D geometry of the world, the robot can recognize a "cup" or a "table" based on its shape and position in space, regardless of the room's lighting or wallpaper.

## 2.2. Main Contributions / Findings
1.  **Development of iDP3:** An improved version of the `3D Diffusion Policy` that works in the "egocentric" (robot's eye view) frame, removing the need for complex camera calibration.
2.  **Whole-Upper-Body Teleoperation:** A system using the Apple Vision Pro to map human movements to the robot's head, waist, arms, and hands, allowing for natural, human-like data collection.
3.  **Single-Scene Training for Multi-Scene Deployment:** Proved that a humanoid can learn a task in one lab setting and successfully perform it in a completely different kitchen or office without further training.
4.  **Rigorous Evaluation:** Conducted over 2253 real-world trials, a significantly higher number than most contemporary humanoid research papers, providing high statistical confidence in the results.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. 3D Point Clouds
A `point cloud` is a set of data points in space, usually defined by `X, Y,` and $Z$ coordinates. Unlike a 2D image (pixels), a point cloud represents the actual geometry of the environment.
*   **Advantage:** It is invariant to color and lighting changes, which is crucial for moving between different rooms.

### 3.1.2. Diffusion Policy (DP)
A `Diffusion Policy` is a type of machine learning model used for robot control. It is based on **Diffusion Models** (the technology behind AI art generators like DALL-E).
*   **Concept:** Instead of trying to predict the exact "best" action directly, the model starts with random noise and iteratively "refines" it into a valid robot action (joint movements) that matches the training data.
*   **Formula (General Denoising):**
    $x_{t-1} = f_\theta(x_t, \text{observation})$
    Where $x$ represents the robot's action sequence, $t$ is the denoising step, and $\theta$ are the learned parameters of the model.

### 3.1.3. Degrees of Freedom (DoF)
`DoF` refers to the number of independent ways a robot can move. A human arm has roughly 7 DoF. This paper uses a 25-DoF robot (head, waist, arms, hands), allowing for very complex, human-like poses.

### 3.1.4. Imitation Learning
This is a training method where a robot learns by "watching" or "mimicking" a human teacher. In this paper, the human uses a VR headset to move the robot, and the model records the relationship between what the robot sees and how the human moves it.

## 3.2. Previous Works
The authors compare their work against several key milestones:
*   **DP3 (3D Diffusion Policy):** The predecessor to this work. It showed that 3D point clouds help robot arms on tabletops, but it required the camera to be in a fixed, known position (`world frame`).
*   **HumanPlus & OmniH2O:** Recent humanoid systems that use whole-body control but struggle with **generalization**—they mostly work in the training lab.
*   **ALOHA:** A popular teleoperation system for dual-arm robots, but it usually relies on 2D vision, making it sensitive to visual changes.

## 3.3. Technological Evolution & Differentiation
The field moved from **Reinforcement Learning** (trial and error in simulation) to **Imitation Learning** (mimicking humans). Within imitation learning, the shift is moving from **2D Vision** (ResNet/Images) to **3D Vision** (Point Clouds).

**This paper's differentiation:**
While prior works like `DP3` were limited to static robot arms on a desk, this paper extends 3D learning to **mobile humanoid robots**. It solves the "calibration problem" by processing 3D data in the robot's own moving camera frame (`egocentric frame`), allowing the robot to walk (or roll) into any room and operate immediately.

---

# 4. Methodology

The system is a "closed-loop" architecture where the robot observes the world, processes the data through a neural network, and outputs motor commands.

## 4.1. Hardware Platform
The robot used is the **Fourier GR1**, a full-sized humanoid. 
*   **Actuators:** 25 DoF utilized (Arms, Hands, Head, Waist).
*   **Vision:** An **Intel RealSense L515 LiDAR** camera is mounted on the head. LiDAR uses lasers to measure distance, creating much more accurate 3D maps than standard webcams.
*   **Base:** For stability during these experiments, the robot is mounted on a height-adjustable cart rather than walking on legs, focusing the research on **manipulation**.

## 4.2. Teleoperation & Data Collection
To train the model, a human must provide "demonstrations."
1.  **Human Sensing:** The operator wears an **Apple Vision Pro**. The headset tracks the human's head and hand positions in 3D space.
2.  **Mapping (Relaxed IK):** The system uses `Inverse Kinematics (IK)` to translate the human's hand position `(x, y, z)` into the specific joint angles the robot needs to move its arm to that same spot.
3.  **Visual Feedback:** The 3D LiDAR data from the robot's head is streamed back to the operator's headset, allowing them to see exactly what the robot sees.

## 4.3. Improved 3D Diffusion Policy (iDP3)
The core contribution is the `iDP3` algorithm. It processes the point cloud and robot's current pose to predict future movements.

### 4.3.1. Egocentric 3D Representation
Standard `DP3` requires "camera calibration"—the robot needs to know exactly where the camera is relative to the floor. In `iDP3`, the authors use the **camera frame**. 
*   **Logic:** The coordinates of every point $(x_i, y_i, z_i)$ are relative to the camera on the robot's head. If the robot moves its head, the whole coordinate system moves with it. This makes the system "plug-and-play" for mobile robots.

### 4.3.2. Scaling Up Vision Input
Previous models used roughly 1,024 points to represent a scene. `iDP3` scales this up to **4,096 points**.
*   **Why?** Since they don't manually "crop" the image to only show the object (to allow for generalization), they need more points to capture the background and the target object simultaneously without losing detail.

### 4.3.3. The Neural Network Architecture
The model uses a **Pyramid Convolutional Encoder** to understand the 3D points. 
1.  **Point Processing:** The points are passed through layers that group nearby points together to understand local shapes (like the edge of a cup).
2.  **Pyramid Features:** It looks at the scene at different scales—both the fine details (the handle) and the global structure (the table).
3.  **Diffusion Backbone:** These visual features are combined with the robot's `proprioception` (current joint positions) and fed into a `Diffusion Transformer`.

    The model predicts a sequence of future actions. The paper found that a **longer prediction horizon** (predicting 16 steps into the future instead of 4) makes the movement much smoother and less "jittery."

The following figure (Figure 2 from the original paper) shows the system architecture:

![该图像是一个示意图，展示了 humanoid 机器人的操作平台、数据采集、学习算法以及部署阶段。左侧介绍了包括 LiDAR 相机和人形机器人 GR1 的平台，中间部分展示了全身机电操作系统获取人类样本数据，右侧则描述了训练于单一场景并能推广到多样化场景的能力。](images/2.jpg)
*该图像是一个示意图，展示了 humanoid 机器人的操作平台、数据采集、学习算法以及部署阶段。左侧介绍了包括 LiDAR 相机和人形机器人 GR1 的平台，中间部分展示了全身机电操作系统获取人类样本数据，右侧则描述了训练于单一场景并能推广到多样化场景的能力。*

---

# 5. Experimental Setup

## 5.1. Tasks
The authors evaluated the robot on three daily tasks:
1.  **Pick & Place:** Grasping a cup and moving it. This tests precision, as humanoid hands are bulky.
2.  **Pour:** Grasping a bottle and pouring its contents into a cup. This tests coordinate movement and orientation.
3.  **Wipe:** Using a sponge to clean a surface. This tests the ability to maintain contact and follow a path.

    The training occurred in a single laboratory setting, but testing occurred in diverse "unseen" settings using different objects.

## 5.2. Evaluation Metrics

### 5.2.1. Success Rate
1.  **Conceptual Definition:** The percentage of attempts where the robot successfully completed the defined task objective (e.g., the cup was moved to the target zone).
2.  **Mathematical Formula:**
    $Success Rate = \frac{N_{success}}{N_{total}} \times 100\%$
3.  **Symbol Explanation:** $N_{success}$ is the count of successful completions; $N_{total}$ is the total number of trials.

### 5.2.2. Efficiency / Smoothness (Attempts per Episode)
1.  **Conceptual Definition:** This measures how "decisive" the policy is. A jittery or confused policy might hover near an object but never try to grab it.
2.  **Standard:** The authors record the number of successful grasps vs. the total number of grasp *attempts*. If the robot attempts many times but fails, the policy is inaccurate. If it doesn't even attempt, the policy is not robust.

## 5.3. Baselines
The authors compared `iDP3` against several strong alternatives:
*   **DP (Diffusion Policy):** The standard 2D version using images.
*   **DP (*R3M):** A 2D version using a pre-trained "robot-specific" visual model (R3M).
*   **Vanilla DP3:** The original 3D model without the authors' improvements for humanoid mobile use.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The primary finding is that **iDP3 generalizes where 2D models fail**.
*   **Scene Generalization:** When moved to a new kitchen, 2D models (DP) were confused by the different colors of the counters. `iDP3`, which sees the world as 3D shapes, recognized the table and cup perfectly.
*   **View Invariance:** If the robot's head was tilted or moved to a different angle than seen in training, the 2D models failed immediately. `iDP3` remained successful (90% success rate) because the 3D geometry of the object relative to the hand didn't change.

## 6.2. Data Presentation (Tables)
The following is Table I from the original paper, comparing the generalization abilities of various humanoid systems:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">Teleoperation</th>
<th colspan="3">Generalization Abilities</th>
<th>Rigorous Policy Evaluation</th>
</tr>
<tr>
<th>Arm &amp; Hand</th>
<th>Head</th>
<th>Waist</th>
<th>Base</th>
<th>Object</th>
<th>Camera View</th>
<th>Scene</th>
<th>Real-World Episodes</th>
</tr>
</thead>
<tbody>
<tr>
<td>AnyTeleop [1]</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>0</td>
</tr>
<tr>
<td>DP3 [2]</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
<td>X</td>
<td>186</td>
</tr>
<tr>
<td>OmniH2O [6]</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>90</td>
</tr>
<tr>
<td>HumanPlus [7]</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>160</td>
</tr>
<tr>
<td>OpenTeleVision [10]</td>
<td>✓</td>
<td>✓</td>
<td>X</td>
<td>X</td>
<td>✓</td>
<td>✓</td>
<td>X</td>
<td>75</td>
</tr>
<tr>
<td><strong>This Work (iDP3)</strong></td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td><strong>2253</strong></td>
</tr>
</tbody>
</table>

The following Table II shows the efficiency of `iDP3` compared to image-based baselines:

<table>
<thead>
<tr>
<th>Baselines</th>
<th>DP</th>
<th>DP3</th>
<th>DP (*R3M)</th>
<th>DP (★R3M)</th>
<th>iDP3 (DP3 Enc)</th>
<th>iDP3</th>
</tr>
</thead>
<tbody>
<tr>
<td>1st-1 (success/attempts)</td>
<td>0/0</td>
<td>0/0</td>
<td>11/33</td>
<td>24/39</td>
<td>15/34</td>
<td>21/38</td>
</tr>
<tr>
<td>1st-2</td>
<td>7/34</td>
<td>0/0</td>
<td>10/28</td>
<td>27/36</td>
<td>12/27</td>
<td>19/30</td>
</tr>
<tr>
<td>3rd-1</td>
<td>7/36</td>
<td>0/0</td>
<td>18/38</td>
<td>26/38</td>
<td>15/32</td>
<td>19/34</td>
</tr>
<tr>
<td>3rd-2</td>
<td>10/36</td>
<td>0/0</td>
<td>23/39</td>
<td>22/34</td>
<td>16/34</td>
<td>16/37</td>
</tr>
<tr>
<td><strong>Total Successes</strong></td>
<td>24/106</td>
<td>0/0</td>
<td>62/138</td>
<td>99/147</td>
<td>58/127</td>
<td>75/139</td>
</tr>
</tbody>
</table>

*(Note: "1st" and "3rd" refer to the number of demonstrations used; "★" indicates a fine-tuned model).*

## 6.3. Ablation Studies
The authors tested what happened if they removed their improvements.
*   **Removing the Pyramid Encoder:** Success dropped because the model couldn't distinguish small objects from the table surface as well.
*   **Reducing Point Count:** If they dropped from 4096 to 1024 points, success rates fell significantly because the "egocentric" view includes too much background noise that overwhelms a sparse point cloud.
*   **Prediction Horizon:** If they predicted only 4 steps (standard), the robot often "froze" or jittered because human demonstration data is noisy. Predicting 16 steps acted as a natural filter, creating smoother motion.

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This work marks a significant milestone in humanoid robotics. It demonstrates that **3D Vision + Diffusion Policies** provide a robust path toward robots that can operate "out of the box" in new environments. By training on just one scene, the Fourier GR1 robot could perform useful tasks in various real-world rooms. The key was moving to an **egocentric 3D representation**, which treats the world as a geometric space rather than a collection of 2D images.

## 7.2. Limitations & Future Work
*   **Lower Body Disconnected:** The robot was on a cart. True "loco-manipulation" (walking while carrying a cup) remains a challenge due to balance hardware limits.
*   **Sensor Noise:** LiDAR sensors are better than RGB cameras for 3D, but they still produce "noise" (phantom points), especially on shiny surfaces like glass or metal.
*   **Teleoperation Fatigue:** Collecting data with the Apple Vision Pro is easier than older methods, but it is still physically tiring for humans, limiting the ability to collect "Big Data."

## 7.3. Personal Insights & Critique
**Inspiration:** 
The transition from "World Frame" to "Egocentric Frame" is a simple but profound shift. It mirrors how biological creatures operate—we don't need to know our exact GPS coordinates to pick up a cup; we only need to know where the cup is relative to our eyes and hands.

**Critique:**
While the results are impressive, the paper notes that a fine-tuned 2D model (`DP ★R3M`) actually outperformed `iDP3` in the *training* environment (99 successes vs 75). This suggests that while 3D is better for **generalization**, 2D vision still holds an edge in pure **precision** for a known environment because 2D images have much higher resolution than sparse 3D point clouds. A future "hybrid" model that uses 3D for spatial awareness and 2D for fine-grained contact might be the ultimate solution for humanoid manipulation.