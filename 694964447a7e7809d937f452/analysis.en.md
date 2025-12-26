# 1. Bibliographic Information

## 1.1. Title
ReBot: Scaling Robot Learning with Real-to-Sim-to-Real Robotic Video Synthesis

## 1.2. Authors
Yu Fang, Yue Yang, Xinghao Zhu, Ka Zhen, Gedas Bertasius, and Danfei Xu. The authors are affiliated with various top-tier research institutions (implied from the context and common collaborations in the robotics field like University of Pennsylvania and University of California, Berkeley).

## 1.3. Journal/Conference
This paper was published on **arXiv** (2025-03-15) as a preprint. ArXiv is a prominent repository for pre-publication research in computer science and robotics, often hosting work that is subsequently presented at top conferences like **CVPR**, **ICRA**, or **CoRL**.

## 1.4. Publication Year
2025

## 1.5. Abstract
The paper addresses the "data scaling" problem in **vision-language-action (VLA)** models. While VLAs show promise by learning directly from real-world data, collecting such data is expensive. The authors propose **ReBot**, a "real-to-sim-to-real" pipeline. It works by replaying real robot movements in a simulator to interact with new virtual objects (**real-to-sim**), then combining these simulated movements with original real-world backgrounds through automated inpainting (**sim-to-real**). This creates physically realistic, temporally consistent videos for training. Experiments show ReBot significantly boosts the performance of models like `Octo` and `OpenVLA` in both simulated environments (SimplerEnv) and real-world Franka robot tasks, outperforming previous generative scaling methods.

## 1.6. Original Source Link
*   **PDF Link:** [https://arxiv.org/pdf/2503.14526v1.pdf](https://arxiv.org/pdf/2503.14526v1.pdf)
*   **Project Page:** [https://yuffish.github.io/rebot/](https://yuffish.github.io/rebot/)

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
The current bottleneck in robot learning is the **data scarcity** of high-quality real-world demonstrations. While models like `Open X-Embodiment` aim to provide large datasets, they are still limited compared to the massive datasets available for pure vision or language models.
*   **The Problem:** Real-world data collection requires physical robots and human operators, making it slow and expensive.
*   **The Gap:** Simulators can generate infinite data, but there is a "sim-to-real gap"—the robot's actions and the camera's observations in a simulator don't look or behave exactly like the real world, causing models trained in sim to fail when deployed on real hardware.
*   **Existing Solutions:** Generative AI (like text-to-video) can create synthetic data, but often produces "hallucinations" (artifacts, inconsistent physics, or objects that disappear/change shape), which confuses the robot's learning process.

## 2.2. Main Contributions / Findings
The paper introduces **ReBot**, which bridges these gaps by:
1.  **Reusing Real Trajectories:** Instead of inventing new movements, it takes successful real-world robot movements and "replays" them in a digital twin environment with new objects.
2.  **Automated Background Integration:** It removes the original robot and object from real videos using AI and replaces them with the simulated counterparts, ensuring the background remains "real" while the manipulation is "new."
3.  **Significant Performance Gains:** ReBot improved the success rate of `OpenVLA` by **21.8%** in simulation and **20%** on a real Franka robot compared to standard pre-trained baselines.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand ReBot, one must be familiar with the following:
*   **Vision-Language-Action (VLA) Models:** These are neural networks that take a camera image (Vision) and a text instruction (Language) to output specific joint movements or gripper positions (Action). Examples include `Octo` and `OpenVLA`.
*   **Digital Twin:** A precise virtual replica of a physical object or environment. In ReBot, the robot and the table are recreated in the `Isaac Sim` software.
*   **Video Inpainting:** A computer vision technique used to remove objects from a video and "fill in" the background so the scene looks natural. ReBot uses a model called `ProPainter` for this.
*   **Segmentation & Tracking:** The ability for an AI to identify the exact pixels belonging to an object (segmentation) and follow those pixels as the object moves (tracking). ReBot uses `GroundedSAM2` for this.

## 3.2. Previous Works
*   **Open X-Embodiment:** A massive collaborative dataset effort to provide diverse robot training data. ReBot uses datasets from this collection (like `BridgeData V2` and `DROID`) as the foundation for its scaling.
*   **ROSIE:** A prior state-of-the-art method that used text-to-image diffusion models to change objects in robotic videos. ReBot critiques ROSIE for lacking **temporal consistency** (the object changes appearance frame-to-frame).
*   **SimplerEnv:** A recently developed simulation platform designed specifically to evaluate how well policies trained on real data perform in a controlled virtual environment.

## 3.3. Technological Evolution
The field has moved from **sim-to-real** (training in sim, testing in real) to **real-to-sim** (modeling real scenes in sim) and now to **generative augmentation**. ReBot represents the next step: **Real-to-Sim-to-Real**, where real data provides the trajectory and background, simulation provides the physics and object diversity, and generative models provide the visual blending.

---

# 4. Methodology

## 4.1. Principles
The core intuition behind ReBot is that a robot's "reaching and grasping" movement for a specific location is often valid for many different objects. By replaying these movements in simulation with different digital assets (e.g., a toy carrot instead of a real spoon), we can create new training samples that are physically grounded but visually diverse.

The following figure (Figure 1 from the original paper) provides an overview of this approach:

![Fig. 1. An overview of ReBot. We propose ReBot, a novel real-tosim-to-real approach for scaling real robot datasets. ReBot replays realworld robot trajectories in a simulation environment to diversify manipulated objects (real-to-sim), and integrates the simulated movements with inpainted real-world background to produce realistic synthetic videos (sim-to-real), effectively adapting VLA models to target domains.](images/1.jpg)
*该图像是一个示意图，展示了ReBot的工作流程。ReBot通过在模拟环境中重放真实机器人的轨迹，实现了真实到模拟（real-to-sim）的回放，并结合真实世界背景进行视频合成，生成真实到模拟再到真实（sim-to-real）的合成视频，有效适配VLA模型到目标领域。*

## 4.2. Core Methodology In-depth

The ReBot pipeline consists of three sequential steps, as illustrated in the following diagram (Figure 2):

![该图像是一个示意图，展示了ReBot方法的三个主要步骤：真实到模拟轨迹重放、真实世界背景修复和模拟到真实视频合成。图中描述了场景解析、动作重放以及合成过程，旨在通过多样化物体操控和生成物理真实的视频以提升机器人学习的效果。](images/2.jpg)
*该图像是一个示意图，展示了ReBot方法的三个主要步骤：真实到模拟轨迹重放、真实世界背景修复和模拟到真实视频合成。图中描述了场景解析、动作重放以及合成过程，旨在通过多样化物体操控和生成物理真实的视频以提升机器人学习的效果。*

### 4.2.1. Step A: Real-to-Sim Trajectory Replay
The goal is to recreate the real-world scene in a simulator (`Isaac Sim`) and replay the movements.

1.  **Scene Parsing and Alignment:** The system identifies the table height and robot pose. It uses `GroundingDINO` to segment the table and calculates the height by processing the point cloud from the camera's depth data.
2.  **Trajectory Replay:** The system extracts the action sequence $\{ \mathbf { a } _ { t } \} _ { t = 1 } ^ { T }$ from the real data.
    *   $t$ represents the timestep.
    *   $T$ is the total number of frames in the episode.
3.  **Object Placement:** To ensure the replay is useful, the virtual object must be placed where the original object was. The system determines $t_{start}$ (when the gripper closes to grasp) and $t_{end}$ (when it opens to release). It calculates the 3D position of the gripper at $t_{start}$ and spawns the new virtual object there.
4.  **Validation:** Not all replays work (e.g., a very large object might slip). ReBot automatically checks the distance between the virtual gripper and the object during the replay. If the object isn't "carried" to the target, the episode is discarded.

### 4.2.2. Step B: Real-world Background Inpainting
To make the final video look real, we need the original background but *without* the original robot and object.

1.  **Segmentation:** The system uses `GroundedSAM2`. It identifies the robot via a text prompt ("robot") and identifies the object by projecting the 3D position calculated in Step A back into the 2D image.
2.  **Removal:** Once the pixels for the robot and object are masked ($m_t$), the model uses `ProPainter` to fill in the background. The result is a "task-agnostic" video sequence $\{ \mathbf { o } _ { t } ^ { \mathrm { r e a l } } \} _ { t = 1 } ^ { T }$ containing just the environment (e.g., the room and the table).

### 4.2.3. Step C: Sim-to-Real Video Synthesis
The final step combines the clean real background with the simulated foreground.

1.  **Merging:** The simulated robot and new object are extracted from the simulation frames $\{ \mathbf { o } _ { t } ^ { \mathrm { s i m } } \} _ { t = 1 } ^ { T }$ and overlaid onto the inpainted real background $\{ \mathbf { o } _ { t } ^ { \mathrm { r e a l } } \} _ { t = 1 } ^ { T }$.
2.  **Dataset Creation:** A new episode $\tau _ { j } ^ { \prime }$ is formed:
    $\tau _ { j } ^ { \prime } = \{ \mathbf { o } _ { t } ^ { \prime } , \mathbf { a } _ { t } , \mathcal { L } ^ { \prime } \} _ { t = 1 } ^ { T }$
    *   $\mathbf { o } _ { t } ^ { \prime }$ is the new synthetic frame.
    *   $\mathbf { a } _ { t }$ is the original real-world action (preserved for physical accuracy).
    *   $\mathcal { L } ^ { \prime }$ is the updated text instruction (e.g., changed from "pick up spoon" to "pick up carrot").

        ---

# 5. Experimental Setup

## 5.1. Datasets
*   **Real Robot Datasets:** `BridgeData V2` (WidowX robot) and `DROID` (Franka Panda robot).
*   **Evaluation Dataset:** The authors collected 220 custom real-world episodes for final Franka robot testing.
*   **Virtual Assets:** Kitchen objects from `Objaverse`, a large library of 3D models.

## 5.2. Evaluation Metrics
1.  **Grasp Rate:** The percentage of trials where the robot successfully closes its gripper on the object.
2.  **Success Rate:** The percentage of trials where the robot completes the entire task (e.g., picking the object up and placing it in a container).
3.  **VBench Scores:** A comprehensive set of metrics for video quality:
    *   **Subject Consistency:** Measures if the object looks the same throughout the video.
    *   **Motion Smoothness:** Measures if the movement looks fluid and jitter-free.
    *   **Imaging Quality:** Measures the sharpness and clarity of the frames.

## 5.3. Baselines
*   **Zero-shot VLA:** Using `Octo` or `OpenVLA` directly without any fine-tuning.
*   **ROSIE:** Scaling the data using a Diffusion-based image inpainting method. This represents the "Generative AI only" approach.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
ReBot outperformed all baselines in both simulation and the real world. A key finding was that ROSIE (the generative baseline) often failed because the objects it "imagined" would change shape or texture between frames, which makes it impossible for a robot controller to track them accurately.

The following figure (Figure 4) quantitatively compares ReBot against ROSIE using VBench:

![Fig. 4. Quantitative comparison of generated video quality. We report VBench scores as evaluation metrics. ReBot outperforms ROSIE and achieves video quality comparable to original real-world videos.](images/4.jpg)
*该图像是图表，展示了生成视频质量的定量比较。以VBench分数作为评价指标，ReBot的表现优于ROSIE，其视频质量接近原始真实视频，包括成像质量、对象一致性、背景一致性和运动平滑度等方面。*

As seen above, ReBot's **Motion Smoothness** ($99.2\%$) and **Subject Consistency** ($95.3\%$) are nearly identical to original real-world videos, while ROSIE lags significantly in consistency.

## 6.2. Data Presentation (Tables)

The following are the results from Table I of the original paper, showing performance on the WidowX robot in simulation:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Put spoon on towel</th>
<th colspan="2">Put carrot on plate</th>
<th colspan="2">Stack green cube on yellow cube</th>
<th colspan="2">Put eggplant in basket</th>
<th colspan="2">Average</th>
</tr>
<tr>
<th>Grasp</th>
<th>Success</th>
<th>Grasp</th>
<th>Success</th>
<th>Grasp</th>
<th>Success</th>
<th>Grasp</th>
<th>Success</th>
<th>Grasp</th>
<th>Success</th>
</tr>
</thead>
<tbody>
<tr>
<td>Octo [29]</td>
<td>34.7%</td>
<td>12.5%</td>
<td>52.8%</td>
<td>8.3%</td>
<td>31.9%</td>
<td>0.0%</td>
<td>66.7%</td>
<td>43.1%</td>
<td>46.5%</td>
<td>16.0%</td>
</tr>
<tr>
<td>Octo+ROSIE [18]</td>
<td>20.8%</td>
<td>2.8%</td>
<td>27.8%</td>
<td>0.0%</td>
<td>18.1%</td>
<td>0.0%</td>
<td>22.3%</td>
<td>0.0%</td>
<td>22.3%</td>
<td>0.7%</td>
</tr>
<tr>
<td>Octo+ReBot (Ours)</td>
<td>61.1%</td>
<td>54.2%</td>
<td>41.1%</td>
<td>22.0%</td>
<td>63.9%</td>
<td>4.2%</td>
<td>52.8%</td>
<td>12.5%</td>
<td>54.7%</td>
<td>23.2%</td>
</tr>
<tr>
<td>OpenVLA [30]</td>
<td>4.2%</td>
<td>0.0%</td>
<td>33.3%</td>
<td>0.0%</td>
<td>12.5%</td>
<td>0.0%</td>
<td>8.3%</td>
<td>4.2%</td>
<td>14.6%</td>
<td>1.1%</td>
</tr>
<tr>
<td>OpenVLA+ROSIE [18]</td>
<td>12.5%</td>
<td>0.0%</td>
<td>41.7%</td>
<td>0.0%</td>
<td>50.0%</td>
<td>0.0%</td>
<td>20.8%</td>
<td>0.0%</td>
<td>31.3%</td>
<td>0.0%</td>
</tr>
<tr>
<td>OpenVLA+ReBot (Ours)</td>
<td>58.3%</td>
<td>20.8%</td>
<td>45.8%</td>
<td>12.5%</td>
<td>66.7%</td>
<td>4.2%</td>
<td>66.7%</td>
<td>54.2%</td>
<td>59.4%</td>
<td>22.9%</td>
</tr>
</tbody>
</table>

The following are the results from Table II of the original paper, showing real-world performance on a Franka Panda robot:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Put carrot in blue plate</th>
<th colspan="2">Put grape in yellow plate</th>
<th colspan="2">Put fanta can in blue plate</th>
<th colspan="2">Put black cube in yellow plate</th>
<th colspan="2">Average</th>
</tr>
<tr>
<th>Grasp</th>
<th>Success</th>
<th>Grasp</th>
<th>Success</th>
<th>Grasp</th>
<th>Success</th>
<th>Grasp</th>
<th>Success</th>
<th>Grasp</th>
<th>Success</th>
</tr>
</thead>
<tbody>
<tr>
<td>Octo [29]</td>
<td>0%</td>
<td>0%</td>
<td>30%</td>
<td>20%</td>
<td>10%</td>
<td>0%</td>
<td>20%</td>
<td>10%</td>
<td>15%</td>
<td>8%</td>
</tr>
<tr>
<td>Octo+ROSIE [18]</td>
<td>30%</td>
<td>20%</td>
<td>0%</td>
<td>0%</td>
<td>20%</td>
<td>20%</td>
<td>10%</td>
<td>0%</td>
<td>15%</td>
<td>10%</td>
</tr>
<tr>
<td>Octo+ReBot (Ours)</td>
<td>40%</td>
<td>20%</td>
<td>40%</td>
<td>30%</td>
<td>30%</td>
<td>20%</td>
<td>30%</td>
<td>30%</td>
<td>35%</td>
<td>25%</td>
</tr>
<tr>
<td>OpenVLA [30]</td>
<td>30%</td>
<td>20%</td>
<td>30%</td>
<td>20%</td>
<td>60%</td>
<td>30%</td>
<td>40%</td>
<td>30%</td>
<td>40%</td>
<td>25%</td>
</tr>
<tr>
<td>OpenVLA+ROSIE [18]</td>
<td>10%</td>
<td>0%</td>
<td>10%</td>
<td>0%</td>
<td>30%</td>
<td>10%</td>
<td>20%</td>
<td>10%</td>
<td>18%</td>
<td>5%</td>
</tr>
<tr>
<td>OpenVLA+ReBot (Ours)</td>
<td>40%</td>
<td>40%</td>
<td>50%</td>
<td>40%</td>
<td>50%</td>
<td>50%</td>
<td>60%</td>
<td>50%</td>
<td>50%</td>
<td>45%</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
*   **Generalization Types:** The authors tested three types of generalization: **Physical** (unseen object sizes), **Semantic** (unseen text instructions), and **Subject** (unseen object categories). ReBot showed improvements in all three, as depicted in Figure 6.
*   **Cross-embodiment:** They tested if data generated for a Franka robot could help a WidowX robot (and vice versa). ReBot demonstrated that its synthetic data is robust enough to help models generalize across different robot hardwares (Figure 7).

    ---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
ReBot effectively solves the "last-mile deployment" problem in robotics by providing a fully automated way to adapt large VLA models to specific new tasks and objects. By combining the **physical reliability of simulation** with the **visual realism of real background inpainting**, it creates a high-quality data pipeline that outperforms purely generative methods.

## 7.2. Limitations & Future Work
*   **Object Diversity:** Currently, ReBot relies on a library of 3D assets (Objaverse). If a task requires a very unique object not in the library, it might still struggle.
*   **Scene Complexity:** The paper primarily focuses on tabletop manipulation. Expanding this to mobile robots or dynamic environments (where the background isn't static) is an area for future research.
*   **Contact Physics:** While Isaac Sim is powerful, simulating complex contact (like tying shoelaces or handling soft fabrics) remains a challenge.

## 7.3. Personal Insights & Critique
ReBot is a clever "best of both worlds" approach. It acknowledges that current generative AI isn't yet good enough to maintain 100% physical consistency in video, so it uses a physics engine as the "anchor." 
**Pros:** The automated validation of trajectories is a vital step—it prevents "garbage in, garbage out" where a model might try to learn from a failed demo.
**Critique:** The reliance on `ProPainter` and `GroundedSAM2` means the system's quality is capped by these underlying AI models. If the inpainting leaves visible "ghosts" or blurry patches, the robot might learn to associate those artifacts with certain actions. However, the results show that even with current inpainting limitations, the performance boost is substantial. This method could potentially be used to "upscale" old robot datasets from 5-10 years ago into modern, high-resolution training data.