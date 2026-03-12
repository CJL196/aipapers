# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is **Vision-Language-Action (VLA) Models**, focusing on their conceptual foundations, technological progress, diverse applications, and significant challenges in the field of embodied artificial intelligence. The full title is: "Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges".

## 1.2. Authors
The authors of this research are **Ranjan Sapkota**, **Yang Cao**, **Konstantinos I. Roumeliotis**, and **Manoj Karkee**. Their research backgrounds and affiliations are diverse and interdisciplinary:
*   **Ranjan Sapkota** and **Manoj Karkee**: Cornell University, Biological & Environmental Engineering, Ithaca, New York, USA.
*   **Yang Cao**: The Hong Kong University of Science and Technology, Department of Computer Science and Engineering, Hong Kong.
*   **Konstantinos I. Roumeliotis**: University of the Peloponnese, Department of Informatics and Telecommunications, Greece.

    This collaboration brings together expertise in engineering, computer science, and robotics, reflecting the cross-disciplinary nature of VLA research.

## 1.3. Journal/Conference
The paper was published on **arXiv** (a preprint server widely used in computer science and physics) on **May 7, 2025**. The identifier is `arXiv:2505.04769v2`. While not a peer-reviewed journal article yet, its presence on arXiv and the comprehensive nature of the review suggest it is intended to be a foundational reference for researchers in the field. The link is provided as `https://arxiv.org/abs/2505.04769`.

## 1.4. Publication Year
The publication year is **2025**. The timestamps indicate a release date of `2025-05-07T19:46:43.000Z`.

## 1.5. Abstract
The abstract summarizes the paper's objective to present a comprehensive synthesis of recent advancements in **Vision-Language-Action (VLA)** models. It outlines five thematic pillars:
1.  **Conceptual Foundations:** Tracing evolution from cross-modal learning to generalist agents.
2.  **Progress:** Covering architectural innovations, training strategies, and inference accelerations (analyzing over 80 models).
3.  **Applications:** Exploring domains like autonomous vehicles, medical/industrial robotics, agriculture, humanoid robots, and AR.
4.  **Challenges:** Analyzing limitations such as data efficiency, safety, and generalization.
5.  **Roadmap:** Proposing a future path where VLAs, Vision-Language Models (VLMs), and Agentic AI converge.
    The project repository is available on GitHub.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2505.04769
*   **PDF Link:** https://arxiv.org/pdf/2505.04769v2
*   **Status:** Preprint (Published on arXiv).

# 2. Executive Summary

## 2.1. Background & Motivation
### Core Problem
The core problem addressed is the historical fragmentation in robotics and artificial intelligence. Prior to the emergence of VLA models, progress happened in isolated domains:
*   **Vision Systems:** Could interpret images (e.g., object detection) but lacked understanding of language or action capabilities.
*   **Language Systems (LLMs):** Could understand/generate text but were restricted to processing language without perceiving the physical world.
*   **Action Systems:** Controlled movement (e.g., robotic arms) but relied on hand-crafted policies that failed to generalize.

    These isolated systems struggled to work together, leading to brittle generalization and labor-intensive engineering when trying to create adaptive behavior in unstructured environments.

### Importance and Gaps
The gap lies in the inability to generate or execute coherent actions based on multi-modal input. While **Vision-Language Models (VLMs)** achieved impressive multi-modal understanding, they lacked the ability to translate this understanding into physical motor control. Existing robots could recognize objects ("apple") or follow text ("pick the apple") separately, but integrating these abilities into fluid, adaptable behavior remained missing. This limitation highlighted a critical bottleneck in **Embodied AI** (AI that exists in a physical body): without systems that jointly perceive, understand, and act, intelligent autonomous behavior remained a challenging goal.

### Innovative Idea
The paper introduces **Vision-Language-Action (VLA)** models as the solution. Conceptualized around 2021-2022 (pioneered by Google DeepMind's Robotic Transformer 2 or RT-2), VLA models unify perception, reasoning, and control within a single framework. They integrate vision inputs, language comprehension, and motor control capabilities, enabling embodied agents to perceive surroundings, understand complex instructions, and execute appropriate actions dynamically.

## 2.2. Main Contributions / Findings
*   **Comprehensive Synthesis:** The paper systematically organizes the landscape of VLA research into five thematic pillars: Concepts, Progress, Applications, Challenges, and Roadmap.
*   **Literature Review:** It adopts a rigorous literature review framework, covering over 80 VLA models published in the past three years (2022-2025).
*   **Technical Taxonomy:** It classifies architectural innovations (e.g., Early Fusion, Dual-System Architectures, Self-Correcting Frameworks) and training efficiency strategies (e.g., Low-Rank Adaptation, Quantization).
*   **Application Mapping:** It grounds developments in real-world domains including humanoid robotics, autonomous vehicles, healthcare, agriculture, and augmented reality navigation.
*   **Forward-Looking Roadmap:** It outlines a future trajectory where VLAs converge with agentic AI to strengthen socially aligned, adaptive, and general-purpose embodied agents.
*   **Repository:** Provides a public project repository for further exploration.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand the VLA framework, several foundational technologies must be defined:

### Vision-Language Models (VLMs)
**VLMs** are AI systems that combine visual encoders (like CNNs or Vision Transformers) with language models. They allow computers to "see" and "read" simultaneously. For example, a VLM can take an image of an orchard and answer questions like "Are the apples ripe?" by matching visual features to linguistic concepts. Common architectures include **CLIP** and **Flamingo**.

### Large Language Models (LLMs)
**LLMs** are neural networks trained on vast amounts of text data. They predict the next word in a sequence, allowing for natural language understanding and generation. Models like **GPT-4** or **Llama** serve as the reasoning backbone for understanding instructions in VLA systems.

### Transformers
The **Transformer** architecture is the underlying engine for modern VLA models. It uses a mechanism called `self-attention` to weigh the importance of different parts of the input data. In VLA, it fuses visual tokens and language tokens into a shared space to enable end-to-end learning.

### Action Tokenization
In VLA models, physical actions (like moving a robotic arm) are converted into discrete symbols called **Action Tokens**. Similar to how words are tokens in language, action tokens represent motor commands (e.g., joint angles, gripper forces). This allows the model to "generate" motion sequences just like it generates text sentences.

### Embodied AI
**Embodied AI** refers to intelligent systems that interact with the physical world through sensors and actuators. Unlike purely digital agents, embodied agents must deal with noise, uncertainty, and physical constraints (e.g., gravity, friction).

## 3.2. Previous Works
The paper traces the evolution through key milestones:
*   **Early Integration (2022-2023):** Models like **CLIPort** combined CLIP embeddings with motion primitives. **RT-1** enabled visual chain-of-thought reasoning. **Gato** demonstrated generalist capabilities across tasks.
*   **Specialization (2024):** Second-generation models like **Deer-VLA** and **ReVLA** incorporated domain-specific biases (e.g., memory efficiency, 3D scene graphs) to handle specific tasks better.
*   **Generalization & Safety (2025):** Recent systems prioritize robustness. **SafeVLA** integrated formal verification for risk-aware decisions. **GR00T N1** implemented dual-system architectures for planning and execution.

## 3.3. Technological Evolution
The technology has evolved from **Isolated Pipelines** (separate modules for vision, language, and control linked manually) to **Unified Agents** (single transformer processing all modalities).
1.  **Phase 1: Modular.** Perception outputs symbolic labels; a planner maps them to actions. High engineering effort, low adaptability.
2.  **Phase 2: VLM-Augmented.** Used pre-trained VLMs for semantic understanding but required separate controllers for action.
3.  **Phase 3: End-to-End VLA.** Vision, language, and action are encoded as tokens and processed together in one network. This enables zero-shot transfer (performing unseen tasks) and compositional reasoning.

## 3.4. Differentiation Analysis
Unlike traditional **Visuomotor Pipelines** which treat perception and control as distinct steps, VLA models support **Semantic Grounding**. They enable context-aware reasoning, affordance detection (understanding what an object allows you to do), and temporal planning. Modern VLAs fuse modalities end-to-end using large-scale pretrained encoders, whereas older systems often relied on hand-engineered interfaces or domain-specific templates.

# 4. Methodology

## 4.1. Principles
The core principle of VLA models is **Unified Token-Based Representation**. Instead of treating vision, language, and action as separate data streams requiring complex post-processing, VLA models convert them all into a shared sequence of tokens. This allows a single autoregressive decoder (similar to how text generation works) to predict the next step in a task sequence. The theoretical basis relies on the assumption that visual, linguistic, and motor patterns share a common latent structure that can be learned jointly from large-scale multimodal data.

## 4.2. Core Methodology In-depth (Layer by Layer)
Since this is a review paper, the "methodology" involves analyzing the standard VLA architecture described in the literature, particularly the tokenization pipeline exemplified in Algorithm 1 of the text. Below is the breakdown of the **VLA Tokenization Pipeline**.

### Step 1: Multimodal Input Acquisition
The system collects three distinct data streams:
1.  **Visual Observations ($I$):** e.g., RGB-D frames (color + depth) from cameras.
2.  **Natural Language Instructions ($T$):** e.g., "Pick the red apple".
3.  **Robot State ($\theta$):** e.g., Joint angles, velocity, proprioceptive signals.

### Step 2: Independent Tokenization
Each stream is encoded independently into compact representations using specialized encoders.

$V \leftarrow \text{ViT}(I)$
Here, $V$ represents the set of visual tokens generated by passing the image $I$ through a Vision Transformer (ViT). The text specifies this produces approximately 400 vision tokens.

$L \leftarrow \text{BERT}(T)$
Here, $L$ represents the sequence of language tokens generated by encoding the text command $T$ using a language model like BERT. This yields approximately 12 semantic language tokens.

$$
S \leftarrow \text{MLP}(\theta)
$$
Here, $S$ is the 64-dimensional state embedding produced by passing the robot state $\theta$ through a Multilayer Perceptron (MLP). This provides real-time awareness of the robot's configuration.

### Step 3: Multimodal Fusion
The individual token sets are fused using a cross-modal attention mechanism to create a shared understanding.

$$
F \leftarrow \text{CrossAttention}(V, L, S)
$$
The result $F$ is a 512-dimensional fused token that captures semantics, intent, and situational awareness needed for grounded action. This step aligns object semantics (from vision), spatial layout, and physical constraints (from state).

### Step 4: Action Prediction
The fused representation is passed to a policy decoder (e.g., a transformer) to generate action tokens.

$A \leftarrow \text{FAST}(F)$
The model predicts a set of action tokens $A$ (approximately 50 tokens). FAST refers to an efficient tokenization scheme discussed in the paper.

### Step 5: Execution
The predicted action tokens are detokenized into continuous motor commands $\tau_{1:N}$ and executed by the robot controller.

The algorithm formalizes this flow as follows:

$$
Algorithm 1 VLA Tokenization Pipeline   

<table><tr><td>1: Input: RGB-D frame I, text command T, joint angles θ</td></tr><tr><td>2: V ← ViT(I) 400 vision tokens 3: L ← BERT(T) &gt; 12 language tokens</td></tr><tr><td>4: S ← MLP(θ) 64-dim state encoding</td></tr><tr><td>5: F ← CrossAttention(V, L, S ) &gt; 512-dim fused token</td></tr><tr><td>6: A ← FAST(F) 50 action tokens</td></tr><tr><td>7: Output: Motor commands T1:N</td></tr></table>
$$

This process mirrors how text generation works in Large Language Models (LLMs), but here the "sentence" is a motion trajectory. The model autoregressively predicts action tokens one step at a time, conditioned on the full multimodal context.

# 5. Experimental Setup

## 5.1. Datasets
Since this is a review paper, it does not conduct experiments itself but analyzes datasets used by the reviewed models. Key datasets mentioned include:
*   **Web-Scale Corpora:** LAION-5B, COCO, HowTo100M. Used for pretraining visual-language alignment.
*   **Robot Trajectory Datasets:**
    *   **Open X-Embodiment (OXE):** Over 4 million robot trajectories across diverse robots.
    *   **RT-X:** Real-robot demonstrations from the RT-1 dataset (over 100,000 demos).
    *   **BridgeData:** Cross-domain datasets to boost generalization.
    *   **Libero:** Specifically designed for long-horizon manipulation tasks.

**Example Data Sample:**
For a typical task in these datasets, the data pair consists of:
*   **Input:** An image of a cluttered tabletop, text "stack the green blocks", and robot joint angles $(\theta)$.
*   **Output:** A sequence of motor commands $(\tau)$ representing the grasp and placement motion.

## 5.2. Evaluation Metrics
The paper reviews evaluation metrics commonly used to benchmark VLA performance:

### Success Rate
1.  **Conceptual Definition:** The percentage of times a robot completes a task correctly given an instruction. It focuses on task completion reliability.
2.  **Mathematical Formula:**
    $$ \text{Success Rate} = \frac{N_{\text{success}}}{N_{\text{total}}} $$
3.  **Symbol Explanation:** $N_{\text{success}}$ is the number of successful trials; $N_{\text{total}}$ is the total number of trials attempted.

### Inference Latency
1.  **Conceptual Definition:** The time taken for the model to generate an action token after receiving input. Crucial for real-time control.
2.  **Mathematical Formula:**
    $$ T_{\text{latency}} = T_{\text{inference}} - T_{\text{input}} $$
3.  **Symbol Explanation:** $T_{\text{inference}}$ is the timestamp when action generation completes; $T_{\text{input}}$ is the timestamp when input data is received.

### Generalization Score
1.  **Conceptual Definition:** Measures performance on unseen objects or environments (zero-shot or few-shot transfer).
2.  **Mathematical Formula:** Often calculated as the drop in success rate between seen (training) and unseen (test) distributions.
3.  **Symbol Explanation:** $\Delta S = S_{\text{seen}} - S_{\text{unseen}}$.

## 5.3. Baselines
The review compares various VLA models against each other rather than fixed baselines, but common benchmarks include:
*   **RT-1 / RT-2:** Early large-scale transformer policies.
*   **CLIPort:** Baseline for visuomotor policies using transporters.
*   **Octo:** Open-source generalist robot policy.
*   **OpenVLA:** Open-source implementation comparable to RT-2.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The review synthesizes results from over 80 models. Key findings include:
*   **Efficiency Gains:** Smaller parameter counts (e.g., 7B parameters in OpenVLA) can outperform larger models (e.g., 55B in RT-2 variants) when co-fine-tuned effectively. Parameter-efficient methods like **Low-Rank Adaptation (LoRA)** reduced trainable weights by up to 70% without performance loss.
*   **Inference Speed:** Advanced acceleration techniques like parallel decoding and compressed action tokens (FAST) achieved up to $2.5\times$ speedups, reducing latency below 5ms for certain high-frequency controls.
*   **Generalization:** Models trained on web-scale data showed improved zero-shot capabilities for novel objects compared to those trained solely on robot data.
*   **Safety:** Integrating formal verification and risk assessment modules significantly reduced unsafe behaviors (by over 80% in some SafeVLA implementations).

## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model Name</th>
<th rowspan="2">Year</th>
<th colspan="2">Architecture Type</th>
<th colspan="2">Policy Emphasis</th>
</tr>
<tr>
<th>End-to-End</th>
<th>Hierarchical</th>
<th>Low-Level Policy</th>
<th>High-Level Planner</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIPort [202]</td>
<td>2022</td>
<td>&nbsp;</td>
<td>&times;</td>
<td>&nbsp;</td>
<td>&times;</td>
</tr>
<tr>
<td>RT-1 [19]</td>
<td>2022</td>
<td>&nbsp;</td>
<td>&times;</td>
<td>&nbsp;</td>
<td>&times;</td>
</tr>
<tr>
<td>Gato [181]</td>
<td>2022</td>
<td>&nbsp;</td>
<td>&times;</td>
<td>&nbsp;</td>
<td>&times;</td>
</tr>
<tr>
<td>VIMA [112]</td>
<td>2022</td>
<td>&nbsp;</td>
<td>&times;</td>
<td>&nbsp;</td>
<td>&times;</td>
</tr>
<tr>
<td>Diffusion Policy [40]</td>
<td>2023</td>
<td>&nbsp;</td>
<td>&times;</td>
<td>&check;</td>
<td>&times;</td>
</tr>
<tr>
<td>ACT [287]</td>
<td>2023</td>
<td>&nbsp;</td>
<td>&times;</td>
<td>&check;</td>
<td>&times;</td>
</tr>
<tr>
<td>OpenVLA [122]</td>
<td>2024</td>
<td>&check;</td>
<td>&times;</td>
<td>&check;</td>
<td>&times;</td>
</tr>
<tr>
<td>CogACT [131]</td>
<td>2024</td>
<td>&times;</td>
<td>&check;</td>
<td>&check;</td>
<td>&check;</td>
</tr>
<tr>
<td>GR00T N1 [14]</td>
<td>2025</td>
<td>&check;</td>
<td>&check;</td>
<td>&check;</td>
<td>&check;</td>
</tr>
</tbody>
</table>

![Figure 1: Evolution from isolated modalities to unified Vision-Language-Action models. Integrated perception, language, and action enable adaptive, generalizable embodied intelligence.](images/1.jpg)  
*该图像是一个示意图，展示了视觉-语言-行为（VLA）模型的整合框架。图中展示了视觉模型理解图像、语言模型理解文本，以及行动模型控制行为的过程。通过耦合的感知-语言-行动机制，机器人能够在开放世界动态下感知、推理与行动。*

Figure 1 illustrates the evolution from isolated modalities to unified VLA models. Integrated perception, language, and action enable adaptive, generalizable embodied intelligence.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th>Model (Ref.)</th>
<th>Architecture (vision / language / action)</th>
<th>Training data</th>
<th>Key strength / uniqueness</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIPort [202]</td>
<td>CLIP-ResNet50 + Transporter-ResNet / CLIP / LingUNet</td>
<td>Self-collected [SC] GPT</td>
<td>Aligns semantic CLIP features with Transporter spatial reasoning for precise SE(2) manipulation.</td>
</tr>
<tr>
<td>RT-2 [299]</td>
<td>ViT-22B or ViT-4B / PaLI-X or PaLM-E / symbol-tuning</td>
<td>VQA + RT-1-Kitchen</td>
<td>Co-finetunes internet-scale VQA with robot data, yielding emergent generalization for embodied tasks.</td>
</tr>
<tr>
<td>Octo [218]</td>
<td>CNN / T5-base / Diffusion Transformer</td>
<td>Open X-Embodiment (OXE)</td>
<td>Large multi-robot policy trained on 4M+ trajectories spanning many robot embodiments.</td>
</tr>
<tr>
<td>OpenVLA [122]</td>
<td>DINOv2 + SigLIP / Prismatic-7B / symbol tuning</td>
<td>OXE + DROID</td>
<td>Open-source RT-2-like VLA; supports efficient LoRA adaptation and broad generalization.</td>
</tr>
<tr>
<td>GR00T N1 [14]</td>
<td>NVIDIA Eagle-2 VLM / Human demos + robot trajectories + planning</td>
<td>Simulation + internet video</td>
<td>Generalist humanoid dual-system design combining planning and diffusion execution for dexterous multi-step control.</td>
</tr>
</tbody>
</table>

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Reference (Year)</th>
<th>VLA methodology</th>
<th>Application area</th>
<th>Strength / key innovation</th>
</tr>
</thead>
<tbody>
<tr>
<td>RoboNurse-VLA [132] (2024)</td>
<td>Vision module (SAM2) + language module (Llama2) with real-time voice-to-action pipeline.</td>
<td>Surgical assistance</td>
<td>Instrument handover with real-time voice cues.</td>
</tr>
<tr>
<td>Mobility VLA [42] (2024)</td>
<td>Hierarchical VLA with long-context VLM for goal localization and topological graph navigation.</td>
<td>Multimodal instruction navigation</td>
<td>Navigation from natural language.</td>
</tr>
<tr>
<td>CoVLA [5] (2025)</td>
<td>CLIP-based vision, Llama-2 language, trajectory prediction for action.</td>
<td>Autonomous driving (dataset + VLA training).</td>
<td>End-to-end autonomous driving.</td>
</tr>
<tr>
<td>ORION [69] (2025)</td>
<td>Hierarchical alignment of 2D/3D visual tokens and language embeddings; autoregressive agent environment—ego modeling.</td>
<td>Holistic end-to-end autonomous driving.</td>
<td>QT-Former for history context, LLM reasoning, generative planner.</td>
</tr>
<tr>
<td>TinyVLA [242] (2025)</td>
<td>Compact multimodal backbone with diffusion-policy decoder.</td>
<td>Fast, data-efficient manipulation control.</td>
<td>Jetsom-class deployment capability.</td>
</tr>
</tbody>
</table>

![Figure 11: Mind-map of application domains for VisionLanguage-Action models, with Humanoid Robotics positioned at the top and remaining domains arranged clockwise to match the order of discussion in this section.](images/11.jpg)  
*该图像是一个示意图，展示了Vision-Language-Action (VLA) 模型的应用领域。中心为“VLA的应用”，周围依次排列着“类人机器人”、“自主车辆系统”、“工业机器人”、“医疗与健康机器人”、“精确与自动化农业”和“互动增强现实导航”。*

Figure 11 displays the application domains of VLA models, including Humanoid Robotics, Autonomous Vehicle Systems, Industrial Robotics, Healthcare, Agriculture, and AR Navigation.

The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Challenge / limitation</th>
<th>Potential solution</th>
<th>Expected impact</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="3"><strong>Real-time inference straints</strong></td>
</tr>
<tr>
<td></td>
<td>Parallel decoding, quantized transformers, and hardware acceleration I (e.g., TensorRT) [129, 122]; reduce autoregressive overhead [75, 142].</td>
<td>Enables real-time control and deployment in latency-critical domains [251, 191] (e.g., UAVs, manipulators).</td>
</tr>
<tr>
<td colspan="3"><strong>Multi-modal action represen- tation</strong></td>
</tr>
<tr>
<td></td>
<td>Hybrid tokenization combining diffusion and autoregressive policies [171]; train on diverse demonstrations and multi-modal outputs [156].</td>
<td>Improves performance on complex, dynamic manipulation with mul- tiple valid solution modes [74].</td>
</tr>
<tr>
<td colspan="3"><strong>Safety assurance in open worlds</strong></td>
</tr>
<tr>
<td></td>
<td>Dynamic risk assessment modules [183, 233]; low-latency emergency-stop and adaptive planning layers [113].</td>
<td>Improves reliability and safety in unpredictable settings (homes, fac- tories, healthcare); enhances user acceptability.</td>
</tr>
<tr>
<td colspan="3"><strong>Dataset bias and grounding</strong></td>
</tr>
<tr>
<td></td>
<td>Curate diverse/debiased datasets [185]; stronger grounding (e.g., CLIP fine-tuning with hard negatives) [282, 17].</td>
<td>Improves fairness and semantic fidelity [109], and enhances general- ization to novel real-world inputs [227, 288, 175].</td>
</tr>
<tr>
<td colspan="3"><strong>Limited 3D perception and reasoning</strong></td>
</tr>
<tr>
<td></td>
<td>clouds with visionlanguage features.</td>
<td>complex environments [129].</td>
</tr>
<tr>
<td colspan="3"><strong>Cross-embodishment general- ization</strong></td>
</tr>
<tr>
<td></td>
<td>Train across diverse morphologies; learn embodiment-agnostic action abstractions; apply cross-domain adaptation [266].</td>
<td>Facilitates policy transfer across robot platforms and configurations [279, 122]." </td>
</tr>
<tr>
<td colspan="3"><strong>Annotation complexity and cost</strong></td>
</tr>
<tr>
<td></td>
<td>Weak supervision, active learning, and synthetic data generation to re- duce manual labeling [148].</td>
<td>Lowers development cost and accelerates scaling to new tasks/domains [233, 286].</td>
</tr>
<tr>
<td colspan="3"><strong>Sim-to-real transfer gap</strong></td>
</tr>
<tr>
<td></td>
<td>Domain adaptation, sim-to-real fine-tuning, and real-world calibration [210, 134].</td>
<td>Improves reliability and consistency when deploying beyond simula- tion [4, 66].</td>
</tr>
<tr>
<td colspan="3"><strong>Ethical and societal implica- tions</strong></td>
</tr>
<tr>
<td></td>
<td>Privacy via on-device processing/anonymization [149, 198, 252, 34]; fairness audits; regulatory and trust frameworks.</td>
<td>Promotes equitable and trustworthy adoption across social, medical, and labor domains [160, 180, 217, 172].</td>
</tr>
</tbody>
</table>

![该图像是一个示意图，展示了视觉-语言-动作模型面临的挑战及其解决方案。挑战包括实时推理、多模态融合安全和系统集成复杂性等；解决方案则涵盖自适应模型剪枝、元学习和领域随机化等策略。](images/17.jpg)  
*该图像是一个示意图，展示了视觉-语言-动作模型面临的挑战及其解决方案。挑战包括实时推理、多模态融合安全和系统集成复杂性等；解决方案则涵盖自适应模型剪枝、元学习和领域随机化等策略。*

Figure 17 illustrates the challenges and solutions associated with Vision-Language-Action models, including real-time inference, multimodal fusion safety, and system integration complexity.

## 6.3. Ablation Studies / Parameter Analysis
While the review paper aggregates results, it highlights specific findings regarding hyper-parameters and components:
*   **Parameter Efficiency:** Reducing trainable weights by 70% using LoRA did not significantly degrade performance on benchmarks, proving that frozen backbones with adapters are effective.
*   **Quantization:** INT8 quantization on embedded platforms preserved approximately 97% of full-precision task success rates, validating its use for edge deployment.
*   **Compression:** Compressed action tokens (FAST) reduced inference latency by up to $15\times$ with minimal trajectory granularity loss.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This comprehensive review evaluates the recent developments, methodologies, and applications of **Vision-Language-Action (VLA)** models. It establishes VLAs as multi-modal systems unifying visual perception, natural language understanding, and action generation. The paper traces their evolution from isolated modules to unified, instruction-following agents. Key contributions include a systematic categorization of over 80 models, an analysis of training efficiency strategies (LoRA, quantization), and a mapping of applications across humanoid robotics, autonomous vehicles, healthcare, and agriculture. The review concludes that while VLAs show strong potential for semantic decision-making, hybrid architectures integrating VLA reasoning with classical controllers remain essential for practical operation due to current limitations in safety and precision.

## 7.2. Limitations & Future Work
### Author-Identified Limitations
1.  **Real-Time Inference:** Autoregressive decoders limit speed, making sub-millisecond updates difficult on edge hardware.
2.  **Action Representation:** Discrete tokenization lacks precision for delicate tasks; continuous approaches suffer from mode collapse.
3.  **Dataset Bias:** Training data often contains stereotypical associations affecting generalization.
4.  **System Complexity:** Integrating high-level planning with low-level control creates synchronization difficulties.

### Future Directions
The authors propose a roadmap converging on **Artificial General Intelligence (AGI)**. Key directions include:
*   **Agentic Learning:** Models that self-propose exploration objectives and continuously adapt.
*   **Neuro-Symbolic Planning:** Combining LLM reasoning with constraint-based verification for safety.
*   **World Models:** Predictive models of dynamics to support counterfactual evaluation.
*   **Governance:** Privacy-preserving inference and ethical alignment frameworks.

## 7.3. Personal Insights & Critique
### Inspirations
The paper provides a crucial taxonomy for a rapidly evolving field. The concept of **Action Tokenization** is particularly inspiring; treating motor commands as a "language" allows the reuse of powerful NLP infrastructure (transformers, attention) for robotics. This abstraction simplifies the interface between high-level cognition and low-level control.

### Potential Issues
1.  **Date Context:** The paper is dated 2025, referencing models that may be speculative or highly advanced projections relative to early 2024 knowledge. Readers should verify the existence of very recent "2025" models cited as published works.
2.  **Hardware Dependency:** Many solutions rely on advanced hardware (e.g., TensorRT, NVIDIA Jetson Orin) which may not be accessible to all researchers, potentially widening the gap between well-funded labs and others.
3.  **Generalization Gap:** Despite claims of zero-shot capability, empirical evidence in the text still notes significant performance drops (up to 40%) when moving to entirely novel task domains, suggesting the "generalist" claim needs cautious interpretation.

### Transferability
The **tokenization pipeline** described in Algorithm 1 could be applied to other embodied domains beyond robotics, such as autonomous drones or virtual avatars, provided the action space can be discretized. The emphasis on **safety-aligned reinforcement learning** is also highly transferable to any safety-critical automated system.

![Figure 18: This conceptual illustration presents "Eva," a future humanoid assistant powered by Vision-Language Models (VLMs), VLA frameworks, and agentic AI systems. VLMs enable semantic scene understanding and object affordance prediction, while VLAs translate language-grounded instructions into hierarchical motor plans. Agentic AI modules ensure adaptive learning, selfrefinement, and interactive decision-making in open-ended environments. Together, these components represent a foundational blueprint for Artificial General Intelligence (AGI) in robotics, where perception, language understanding, planning, and safe autonomous behavior converge in real-world, socially aware tasks.](images/18.jpg)  
*该图像是一个示意图，展示了未来的智能助手"Eva"，该助手融合了视觉语言模型（VLM）和视觉语言行动框架（VLA）。VLM用于语义场景理解和对象预测，而VLA将语言指令转换为分层运动计划，表示人工通用智能（AGI）的基础蓝图。*

Figure 18 presents a conceptual illustration of "Eva," a future humanoid assistant powered by VLMs, VLA frameworks, and Agentic AI, representing a blueprint for AGI in robotics.

![该图像是一个示意图，展示了未来的视觉-语言-行动(VLA)模型发展路线图，包括高效部署、可靠与安全智能、统一系统与治理等三大主题。图中列出了多项关键技术和策略，如紧凑的动作标记化与分块、稳健的多模态基础、物理与因果预测模型等。](images/19.jpg)  
*该图像是一个示意图，展示了未来的视觉-语言-行动(VLA)模型发展路线图，包括高效部署、可靠与安全智能、统一系统与治理等三大主题。图中列出了多项关键技术和策略，如紧凑的动作标记化与分块、稳健的多模态基础、物理与因果预测模型等。*

Figure 19 outlines the future roadmap for VLA models, emphasizing Efficient Deployment, Reliable & Safe Intelligence, and Unified Systems & Governance.