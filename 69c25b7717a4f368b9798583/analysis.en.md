# 1. Bibliographic Information
## 1.1. Title
The paper focuses on **Physically Plausible Video Generation (PPVG)**, with the core proposal of an event-centric causal reasoning framework that models physical phenomena as sequences of causally connected dynamic events to generate videos that follow real-world physical laws.
## 1.2. Authors
The authors and their affiliations are as follows:
- Zixuan Wang*, Yixin Hu*, Haolan Wang, Yinjie Lei: Sichuan University, Chengdu, China
- Feng Chen: The University of Adelaide, Adelaide, Australia
- Yan Liu: Hong Kong Polytechnic University, Hong Kong, China
- Wen Li: University of Electronic Science and Technology of China, Chengdu, China
  The lead authors are early-career researchers specializing in computer vision, generative models, and physical reasoning.
## 1.3. Publication Venue & Status
This paper is published as a preprint on arXiv, the largest open-access preprint repository for computer science and physics research. Preprints on arXiv are widely circulated and cited in the computer vision community, and this work is currently under review for top computer vision conferences (e.g., CVPR, ICCV).
## 1.4. Publication Year
Published on March 10, 2026 (UTC time).
## 1.5. Abstract
This work addresses the core limitation of existing video diffusion models: they often render physical phenomena as static single moments, failing to model the causal temporal progression of real-world physical processes. The authors propose a two-module framework:
1.  **Physics-driven Event Chain Reasoning (PECR):** Decomposes the physical phenomenon described in user prompts into ordered elementary event units, using embedded physical formulas as constraints to eliminate causal ambiguity in reasoning.
2.  **Transition-aware Cross-modal Prompting (TCP):** Converts event units into temporally aligned text-visual prompts, ensuring causal coherence and visual continuity between events via progressive narrative revision and interactive keyframe synthesis.
    Comprehensive experiments on two standard PPVG benchmarks (PhyGenBench, VideoPhy) show the framework outperforms all existing state-of-the-art methods by a significant margin.
## 1.6. Original Source Links
- Abstract page: https://arxiv.org/abs/2603.09094
- Full PDF: https://arxiv.org/pdf/2603.09094v1
- Publication status: Public preprint, not yet officially published in a peer-reviewed conference/journal.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem to Solve
Existing video generation models (including state-of-the-art systems like OpenAI Sora, Kling) can generate photorealistic videos, but they frequently produce physically impossible content (e.g., ice melting into fire, objects floating without force, light passing through opaque objects) because they lack explicit modeling of physical laws and causal event progression.
### Importance of the Problem
PPVG is critical for high-impact real-world applications:
- Movie production: Generating realistic special effects that follow physical laws
- Autonomous driving: Synthesizing edge-case driving scenarios for model training
- Embodied AI: Simulating real-world physical interactions for robot manipulation training
### Gaps in Prior Research
Existing PPVG methods use large language models (LLMs) to add physical concept tags to user prompts, but they have two key limitations:
1.  **Causal ambiguity:** They treat physical phenomena as static single moments, not as ordered sequences of causally connected events
2.  **Insufficient continuous constraints:** Static text prompts cannot convey the smooth temporal transitions between physical events, and aligned visual priors for specific physical processes are hard to obtain
### Innovative Entry Point
The authors reframe PPVG as the task of generating a sequence of causally linked, dynamically evolving events, rather than generating a static scene matching a text prompt. They use explicit physical formula constraints to eliminate reasoning ambiguity, and progressive cross-modal (text + visual) prompts to ensure temporal continuity between events.
## 2.2. Main Contributions / Findings
The paper's primary contributions are:
1.  A novel event-centric PPVG framework that explicitly models the causal progression of physical phenomena
2.  The PECR module, which decomposes complex physical phenomena into ordered event units using physical formula constraints to ensure deterministic causal dependencies
3.  The TCP module, which generates temporally aligned text-visual prompts to maintain causal coherence and visual continuity between events
4.  State-of-the-art performance on two standard PPVG benchmarks: 8.19% higher average score than prior SOTA on PhyGenBench, and 3.4% higher combined semantic-physical accuracy on VideoPhy
5.  Extensive ablation studies proving the effectiveness of each component of the framework, with the interactive keyframe synthesis component being the most critical for performance.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
We define all core technical terms required to understand this paper for beginners:
1.  **Physically Plausible Video Generation (PPVG):** A subfield of video generation that requires generated videos to strictly follow real-world physical laws (e.g., Newtonian mechanics, thermodynamics, optics) and causal logic, in addition to being photorealistic and semantically aligned with prompts.
2.  **Video Diffusion Model:** The dominant current video generation paradigm, which works in two phases:
    - Forward phase: Gradually add Gaussian noise to real videos until only random noise remains
    - Reverse phase: Train a neural network to gradually denoise random noise into realistic videos, conditioned on text prompts or other signals
3.  **Chain-of-Thought (CoT) Reasoning:** A prompt engineering technique for LLMs that asks the model to output intermediate reasoning steps before giving the final answer, significantly improving its performance on complex reasoning tasks (e.g., mathematical calculation, physical reasoning).
4.  **Scene Graph:** A structured representation of a visual scene, where nodes represent objects (with their attributes like color, size, material) and edges represent relationships between objects (e.g., "A is above B", "A collides with B").
5.  **Cross-Modal Prompting:** Using multiple types of input signals (e.g., text + images) as conditioning for generative models, to provide more complete constraints than single-modality (text-only) prompts.
6.  **Keyframe Interpolation:** A video processing technique that generates intermediate frames between two manually or automatically generated keyframes, to create smooth continuous motion between the keyframes.
## 3.2. Previous Works
The paper categorizes prior relevant research into three core areas:
### 1. Physically Plausible Video Generation
Prior PPVG methods fall into two categories:
- **Simulation-based methods:** Integrate graphics physics engines (e.g., Taichi, DiffTaichi) into the diffusion sampling process to enforce physical constraints. These methods produce highly accurate physical results but require manual specification of engine parameters for each scenario, and cannot handle open-domain physical phenomena.
- **LLM prompt-based methods:** Use LLMs to add physical concept tags to user prompts (e.g., adding "viscous liquid, slow flow" to a prompt about pouring honey) to guide diffusion models to generate more physically plausible content. Examples include PhyT2V [6], DiffPhy [8], and PhysHPO [16]. These methods handle open-domain scenarios but do not model the causal temporal progression of physical events.
### 2. Chain-of-Thought in Visual Generation
Recent works have adapted CoT reasoning from NLP to visual generation, falling into two paradigms:
- **Reasoning before generation:** Use LLMs to refine user prompts into more detailed, fine-grained descriptions before starting the generation process, e.g., IRG [19], Draw-In-Mind [20]. These methods focus on spatial and semantic detail, not causal temporal progression.
- **Reasoning during generation:** Embed step-by-step reasoning into the generation process, e.g., Z-Sampling [23], Visual-CoG [24]. These methods improve semantic consistency across frames but do not enforce physical law constraints.
### 3. Dual-Prompt in Video Generation
To address the limitation that text prompts underspecify geometric and motion details, recent works add visual cues (reference images, sketches, motion trajectories) as additional conditioning for video generation. Examples include SketchVideo [30] (uses sketches to constrain object contours), TrackGo [33] (uses motion trajectories to constrain object movement). These methods constrain individual scenes but do not ensure smooth transitions between multiple consecutive events.
## 3.3. Technological Evolution
The evolution of PPVG technology has followed three stages:
1.  **Stage 1 (2019-2023):** Simulation-integrated diffusion models, which are accurate but limited to closed-domain, manually parameterized scenarios
2.  **Stage 2 (2024-2025):** LLM-augmented prompt-based methods, which handle open-domain scenarios but do not model causal event progression
3.  **Stage 3 (2026 onwards, this work):** Event-centric causal reasoning frameworks, which combine physical law constraints, causal event decomposition, and cross-modal progressive prompts to generate open-domain, physically plausible videos with correct temporal progression
    This work represents the first successful implementation of the stage 3 paradigm for PPVG.
## 3.4. Differentiation Analysis
Compared to all prior PPVG methods, this work has three core unique innovations:
1.  **Event-centric modeling:** Unlike prior methods that model physical phenomena as static single moments, this work decomposes phenomena into ordered sequences of causally connected events, explicitly modeling temporal progression.
2.  **Physical formula constraints:** Unlike prior methods that use only qualitative semantic physical tags, this work embeds quantitative physical formulas into the reasoning process to eliminate causal ambiguity and ensure deterministic physical logic.
3.  **Progressive cross-modal prompts:** Unlike prior methods that use static text-only or static text+image prompts, this work dynamically generates evolving text and visual keyframe prompts for each event, ensuring smooth transitions between consecutive physical phases.

# 4. Methodology
## 4.1. Principles
The core guiding principle of this work is that all real-world physical phenomena are ordered sequences of causally linked events, where each event is determined by the previous event and physical laws. To generate physically plausible videos, the framework must:
1.  First decompose the user's text description of a physical phenomenon into a logically ordered sequence of event units, using explicit physical formula constraints to ensure no causal ambiguity
2.  Then convert these event units into temporally aligned text and visual prompts that evolve with the physical process, to guide the diffusion model to generate smooth, continuous, physically consistent video content between events
## 4.2. Core Methodology In-depth
The overall framework follows the mapping function:
$\Gamma : w \to \mathbf { V }$
Where $w$ is the user-provided text description of a physical phenomenon, $\mathbf{V}$ is the output physically plausible video, and $\Gamma$ is the end-to-end framework consisting of two synergistic modules: Physics-driven Event Chain Reasoning (PECR) and Transition-aware Cross-modal Prompting (TCP). We explain each module step-by-step below, integrating all original formulas exactly as presented in the paper.
### 4.2.1. Physics-driven Event Chain Reasoning (PECR) Module
This module converts the user's text prompt into an ordered sequence of physical event units, with two core components: Physics Formula Grounding and Physical Phenomena Decomposition.
#### Step 1: Physics Formula Grounding
First, we identify the physical laws described in the user prompt $w$ via LLM question answering, following the classification defined in [35]. We then retrieve the corresponding physical formulas from an online physics knowledge base, using the formula names associated with the identified physical laws as queries. The retrieval formula is:
$$
\mathcal { F } ^ { * } = \mathrm { T o p K } _ { f \in \mathcal { F } _ { \mathcal { L } } } P ( f \mid \mathcal { N } _ { \mathcal { L } } , \mathcal { L } )
$$
Where:
- $\mathcal{L}$ = the physical law identified from the user prompt
- $\mathcal{N_L}$ = the name of the physical formula associated with $\mathcal{L}$
- $\mathcal{F_L}$ = all formulas in the knowledge base related to physical law $\mathcal{L}$
- $P(\cdot)$ = the scoring function that ranks candidate formulas by relevance to $\mathcal{N_L}$ and $\mathcal{L}$
- $\mathcal{F}^*$ = the top-K most relevant retrieved physical formulas
  If no direct match for $\mathcal{N_L}$ is found in the knowledge base, the LLM is used to regenerate formula names based on $\mathcal{F_L}$. Once the correct formula is retrieved, the LLM infers the values of all physical parameters required for the formula via commonsense reasoning.
The following figure (Figure 2 from the original paper) illustrates this process for an example of oil-water volume change:

![该图像是示意图，展示了物理公式的基础知识与现象分解过程。左侧通过文本提示引入物理现象，并利用体积守恒定律 $A_1h_1=A_2h_2$ 提取相关参数。右侧则展示了事件分解，推导出物理条件和场景图，从而描述油的高度变化及其动态关系。](images/2.jpg)
*该图像是示意图，展示了物理公式的基础知识与现象分解过程。左侧通过文本提示引入物理现象，并利用体积守恒定律 $A_1h_1=A_2h_2$ 提取相关参数。右侧则展示了事件分解，推导出物理条件和场景图，从而描述油的高度变化及其动态关系。*

#### Step 2: Physical Phenomena Decomposition
We decompose the physical phenomenon into an ordered sequence of event units:
$$
\{ \mathcal { E } _ { t } \} _ { t = 1 } ^ { T } = \{ \{ \mathcal { C } _ { t } \} _ { t = 1 } ^ { T } , \{ \mathcal { G } _ { t } \} _ { t = 1 } ^ { T } \}
$$
Where:
- $T$ = total number of decomposed events
- $\mathcal{C}_t$ = physical conditions for the $t$-th event, consisting of physical parameter values and their calculated results from the retrieved formula
- $\mathcal{G}_t$ = dynamic scene graph for the $t$-th event, describing objects and their interactions
  First, we calculate physical conditions $\mathcal{C}_t$ using the retrieved formula $\mathcal{F}^*$. We define a new event as occurring when the physical parameters change by more than a predefined threshold $\tau_p$:
$$
\mathcal { C } _ { t } = \left\{ \left( \mathbf { P } _ { t } , \mathcal { F } ^ { * } ( \mathbf { P } _ { t } ) \right) \big | \| \mathbf { P } _ { t } - \mathbf { P } _ { t - 1 } \| > \tau _ { p } \right\}
$$
Where:
- $\mathbf{P}_t$ = vector of physical parameters for all objects in the $t$-th event
- $\tau_p$ = threshold for parameter change, tuned to 0.1 in all experiments
- $\|\cdot\|$ = L2 norm of the parameter difference vector
  To ensure physical consistency, we validate parameters for each event against neighboring events: if an abrupt parameter change violates physical continuity, the parameters are sent back to the LLM for re-inference.
Next, we update the scene graph $\mathcal{G}_t$ based on the calculated physical conditions $\mathcal{C}_t$:
$$
\mathcal { G } _ { t } = \Phi ( \mathcal { G } _ { t - 1 } , \mathcal { C } _ { t } )
$$
Where:
- $\mathcal{G}_{t-1}$ = scene graph from the previous event
- $\Phi(\cdot)$ = scene graph update function
  The update function modifies two parts of the scene graph:
1.  Nodes: Update object attributes (e.g., appearance, semantic label) based on changes in their physical parameters (e.g., ice melting into water changes the object's material label from "solid ice" to "liquid water")
2.  Edges: Update interaction relationships between objects (e.g., "A is far from B" changes to "A collides with B") based on coordinated parameter changes across multiple objects.
    The overall workflow of the PECR module is shown in Figure 1 from the original paper:

    ![该图像是示意图，展示了物理驱动事件链推理和转移感知的跨模态提示模块的工作流程。图中包含物理公式 $V = A imes h$，并描述了油与水的交互过程，强调了事件之间的因果关系和动态演变。](images/1.jpg)
    *该图像是示意图，展示了物理驱动事件链推理和转移感知的跨模态提示模块的工作流程。图中包含物理公式 $V = A imes h$，并描述了油与水的交互过程，强调了事件之间的因果关系和动态演变。*

### 4.2.2. Transition-aware Cross-modal Prompting (TCP) Module
This module converts the event sequence from PECR into temporally aligned cross-modal prompts to guide video diffusion generation, with two core components: Progressive Narrative Revision and Interactive Keyframe Synthesis. The video diffusion generation step follows the standard formulation:
$$
\mathbf { Z } _ { \tau _ { z } - 1 } = \epsilon _ { \theta } ( \mathbf { Z } _ { \tau _ { z } } ; \mathbf { W } )
$$
Where:
- $\mathbf{Z}_{\tau_z}$ = noised visual prior at diffusion timestep $\tau_z$
- $\mathbf{W}$ = embedded cross-modal prompt
- $\epsilon_\theta$ = video diffusion denoising network
#### Step 1: Progressive Narrative Revision
To avoid semantic redundancy and ensure narrative coherence across events, we generate event descriptions incrementally instead of describing each event independently:
$$
w _ { t } = \mathrm { L L M } \big ( w _ { t - 1 } + \Delta ( w _ { t - 1 } , \mathcal { C } _ { t } , \mathcal { G } _ { t } ) \big )
$$
Where:
- $w_{t-1}$ = description of the previous event
- $\Delta(\cdot)$ = incremental semantic revision function, which adds only the changes in physical conditions $\mathcal{C}_t$ and scene graph $\mathcal{G}_t$ to the previous description
- $w_t$ = description of the current event
  Since video diffusion models are conditioned on a single text prompt, we merge all event descriptions into a single positive semantic prompt $w_+^*$ using causal connectives (e.g., "then", "after that", "as a result") to eliminate redundancy. We also construct a negative prompt $w_-^*$ that describes physically impossible transitions (e.g., "ice does not melt into fire", "objects do not float in air without force"). The final text embedding is:
$$
\mathbf { W } = [ \psi _ { \mathrm { t e x t } } ( w _ { + } ^ { * } ) ; ~ \psi _ { \mathrm { t e x t } } ( w _ { - } ^ { * } ) ]
$$
Where $\psi_{\text{text}}(\cdot)$ is the pre-trained text encoder of the diffusion model, and `[;]` denotes concatenation of the positive and negative embeddings.
#### Step 2: Interactive Keyframe Synthesis
Text prompts are inherently ambiguous for specifying visual details of physical transitions, so we synthesize visual keyframes for each event to provide explicit visual priors for the diffusion model. First, we infer the editing operator required to convert the keyframe of the previous event to the keyframe of the current event:
$$
\mathcal { O } _ { t } = \mathrm { L L M } \big ( ( \mathcal { C } _ { t - 1 } , \mathcal { G } _ { t - 1 } ) \to ( \mathcal { C } _ { t } , \mathcal { G } _ { t } ) \big )
$$
Where $\mathcal{O}_t$ is the editing operator (e.g., "drag the ice cube down by 2cm", "mask the left half of the ice and change it to water"). The keyframe for the current event is generated by applying this operator to the previous keyframe using Qwen-Image-Edit [36]:
$$
v _ { t } = \mathrm { E d i t } ( v _ { t - 1 } ; \mathcal { O } _ { t } )
$$
The first keyframe $v_1$ is directly synthesized from the description of the first event.
Next, we interpolate intermediate frames between consecutive keyframes to ensure smooth transitions. The LLM first predicts the physically plausible time span $d_t$ for the transition between event `t-1` and event $t$, which determines the number of intermediate frames to generate. We encode each keyframe using the VAE encoder of the diffusion model, then perform linear interpolation between the encoded keyframe features:
$$
\begin{array} { r } { \mathbf { z } _ { 0 , t } = \mathrm { I N T E R P } ( \psi _ { \mathrm { i m g } } ( v _ { t - 1 } ) , \ \psi _ { \mathrm { i m g } } ( v _ { t } ) ; \ d _ { t } ) , } \end{array}
$$
Where:
- $\psi_{\text{img}}(\cdot)$ = VAE encoder of the diffusion model
- $\mathrm{INTERP}(\cdot)$ = linear interpolation function
- $\mathbf{z}_{0,t}$ = interpolated latent features for the transition between event `t-1` and $t$
  We concatenate all interpolated latent features to form the full latent sequence $\mathbf{z}_0, \mathbf{z}_1, ..., \mathbf{z}_T$, then add Gaussian noise to get the initial noised prior for the diffusion process:
$$
{ \mathbf Z } _ { \tau _ { z } } = [ { \mathbf z } _ { 0 } , \ldots , { \mathbf z } _ { T } ] + \sigma _ { \tau _ { z } } ^ { 2 } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I )
$$
Where $\sigma_{\tau_z}^2$ is the noise variance corresponding to diffusion timestep $\tau_z$, set to 0.3 in all experiments.
The overall workflow of the TCP module is shown in Figure 3 from the original paper:

![该图像是示意图，展示了用于生成物理可行视频的两大关键模块：逐步叙述修订和交互式关键帧合成。图中左侧描述了如何生成事件描述并总结成语义提示；右侧展示了通过文本编码器与视频扩散框架进行关键帧插值和噪声添加的流程。这些步骤共同促进了物理现象的动态因果视频生成。](images/3.jpg)
*该图像是示意图，展示了用于生成物理可行视频的两大关键模块：逐步叙述修订和交互式关键帧合成。图中左侧描述了如何生成事件描述并总结成语义提示；右侧展示了通过文本编码器与视频扩散框架进行关键帧插值和噪声添加的流程。这些步骤共同促进了物理现象的动态因果视频生成。*

# 5. Experimental Setup
## 5.1. Datasets
The authors evaluate the framework on two standard, widely used PPVG benchmarks:
### 1. PhyGenBench [9]
- Source: Published in 2024, a dedicated benchmark for PPVG evaluation
- Scale: 160 expert-designed text prompts, covering 27 physical laws across 4 core physical domains:
  1.  Mechanics (e.g., object collision, gravity, spring compression)
  2.  Optics (e.g., light refraction, reflection, shadow formation)
  3.  Thermal (e.g., ice melting, water boiling, object combustion)
  4.  Material (e.g., butter spreading, glass breaking, cloth tearing)
- Example prompt: "A cube of ice is placed in a glass of hot water, gradually melting into water as the temperature equalizes"
- Rationale for selection: It is the most comprehensive benchmark for evaluating cross-domain physical plausibility in video generation, with standardized evaluation metrics accepted by the research community.
### 2. VideoPhy [10]
- Source: Published in 2024, a benchmark for evaluating physical commonsense in video generation
- Scale: 688 human-verified text prompts, covering three types of physical interactions:
  1.  Solid-solid (e.g., knife cutting bread, ball hitting a wall)
  2.  Solid-fluid (e.g., honey pouring into a cup, stone sinking in water)
  3.  Fluid-fluid (e.g., oil mixing with water, ink diffusing in water)
- Example prompt: "A knife spreads soft butter evenly on a slice of wheat bread"
- Rationale for selection: It focuses on evaluating physical interactions between objects, which is a key challenge for existing video generation models.
## 5.2. Evaluation Metrics
All metrics are explained following the required three-part structure:
### 1. Physical Commonsense Alignment (PCA) [PhyGenBench]
#### Conceptual Definition
PCA is the primary metric for PhyGenBench, measuring the overall physical plausibility of generated videos by averaging three sub-scores:
- Phenomena Detection (PD): Whether the target physical phenomenon described in the prompt appears in the video
- Physical Order (PO): Whether the sequence of events in the video follows the correct causal order dictated by physical laws
- Naturalness (N): Whether the video is visually realistic and free of artifacts
#### Mathematical Formula
$$
\text{PCA} = \frac{1}{3} \times (\text{PD} + \text{PO} + \text{N})
$$
#### Symbol Explanation
- $\text{PD} \in [0,1]$: Binary score, 1 if the target phenomenon is present, 0 otherwise
- $\text{PO} \in [0,1]$: Binary score, 1 if the event order is physically correct, 0 otherwise
- $\text{N} \in [0,1]$: Continuous score rated by human evaluators or a pre-trained evaluator model, 1 for perfectly natural content
### 2. Semantic Adherence (SA) [VideoPhy]
#### Conceptual Definition
SA measures whether the generated video semantically matches the user prompt, verifying that all entities, actions, and attributes described in the prompt are present in the video.
#### Mathematical Formula
$$
\text{SA} = \begin{cases} 
1 & \text{if all semantic elements in the prompt are present in the video} \\
0 & \text{otherwise}
\end{cases}
$$
#### Symbol Explanation
Semantic elements include all objects, actions, and attributes explicitly mentioned in the prompt (e.g., for the prompt "knife spreading butter", the elements are "knife", "butter", "spreading action").
### 3. Physical Commonsense (PC) [VideoPhy]
#### Conceptual Definition
PC measures whether the generated video follows real-world physical laws, penalizing physically impossible content (e.g., objects floating without force, liquids flowing upward).
#### Mathematical Formula
$$
\text{PC} = \begin{cases} 
1 & \text{if no physical law violations are observed in the video} \\
0 & \text{otherwise}
\end{cases}
$$
#### Symbol Explanation
Physical law violations are detected by the pre-trained VideoCon-Physics evaluator [10], which is trained on thousands of real and synthetic physical videos to identify impossible physical behavior.
## 5.3. Baselines
The authors compare their method against two categories of representative baselines:
### 1. General Video Foundation Models
These are state-of-the-art open-source and commercial general video generation models, including:
- Lavie [40], VideoCrafter v2.0 [41], Open-Sora v1.2 [42], Vchitect v2.0 [43], Wan [44], Kling [45], Pika [46], Gen-3 [47], CogVideoX-5B [38]
- Rationale for selection: These represent the current best performance for general photorealistic video generation, so they provide a benchmark for the physical plausibility of non-specialized models.
### 2. Physics-Aware Video Generation Models
These are the latest specialized PPVG methods, including:
- WISA [15], DiffPhy [8], +PhyT2V [6], +SGD [7], +PhysHPO [16]
- Rationale for selection: These are the previous state-of-the-art methods for PPVG, so comparison against them validates the improvement of the proposed framework over existing specialized methods.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The proposed framework achieves consistent state-of-the-art performance across all benchmarks:
1.  **PhyGenBench:** The framework achieves an average PCA score of 0.66, outperforming the previous SOTA method PhysHPO by 8.19% (relative improvement). It achieves the highest score in all four physical domains (mechanics, optics, thermal, material).
2.  **VideoPhy:** The framework achieves a 49.3% combined SA&PC=1 score (percentage of videos that are both semantically correct and physically plausible), outperforming the previous SOTA method PhysHPO by approximately 3.4% (relative improvement). It also outperforms all baselines in all three interaction categories (solid-solid, solid-fluid, fluid-fluid).
    The qualitative results below (Figure 4 from the original paper) compare the proposed method against the CogVideoX-5B baseline on four physical phenomena:

    ![该图像是一个示意图，展示了四个物理现象的生成比较，包括玻璃球下沉、动态照明、冰融化和火焰传播。左侧为基线生成，右侧为我们的方法生成，展示了在不同条件下的物理效果。](images/4.jpg)
    *该图像是一个示意图，展示了四个物理现象的生成比较，包括玻璃球下沉、动态照明、冰融化和火焰传播。左侧为基线生成，右侧为我们的方法生成，展示了在不同条件下的物理效果。*

The baseline generates static, physically incorrect content (e.g., ice does not melt gradually, fire appears instantaneously), while the proposed method generates smooth, physically correct temporal progression of the phenomena.
Additional qualitative results on VideoPhy are shown in Figure 5 from the original paper:

![该图像是插图，展示了三种物理现象的生成过程，分别是刀具在面包上涂抹黄油、蜜蜂倒入茶杯以及重物压缩弹簧垫。每种现象都有基线与我们的方法对比，强调了连续性与因果关系的表现。](images/5.jpg)
*该图像是插图，展示了三种物理现象的生成过程，分别是刀具在面包上涂抹黄油、蜜蜂倒入茶杯以及重物压缩弹簧垫。每种现象都有基线与我们的方法对比，强调了连续性与因果关系的表现。*

The baseline generates physically incorrect behavior (e.g., honey is not viscous, spring does not compress monotonically), while the proposed method generates correct physical interactions between objects.
## 6.2. Data Presentation (Tables)
The following are the results from Table 1 of the original paper, showing performance on PhyGenBench across four physical domains:

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th colspan="4">Physical domains (↑)</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>Mechanics</th>
<th>Optics</th>
<th>Thermal</th>
<th>Material</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6">Video Foundation Model</td>
</tr>
<tr>
<td>Lavie [40]</td>
<td>0.30</td>
<td>0.44</td>
<td>0.38</td>
<td>0.32</td>
<td>0.36</td>
</tr>
<tr>
<td>VideoCrafter v2.0 [41]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.48</td>
</tr>
<tr>
<td>Open-Sora v1.2 [42]</td>
<td>0.43</td>
<td>0.50</td>
<td>0.34</td>
<td>0.37</td>
<td>0.44</td>
</tr>
<tr>
<td>Vchitect v2.0 [43]</td>
<td>0.41</td>
<td>0.56</td>
<td>0.44</td>
<td>0.37</td>
<td>0.45</td>
</tr>
<tr>
<td>Wan [44]</td>
<td>0.36</td>
<td>0.53</td>
<td>0.36</td>
<td>0.33</td>
<td>0.40</td>
</tr>
<tr>
<td>Kling [45]</td>
<td>0.45</td>
<td>0.58</td>
<td>0.50</td>
<td>0.40</td>
<td>0.49</td>
</tr>
<tr>
<td>Pika [46]</td>
<td>0.35</td>
<td>0.56</td>
<td>0.43</td>
<td>0.39</td>
<td>0.44</td>
</tr>
<tr>
<td>Gen-3 [47]</td>
<td>0.45</td>
<td>0.57</td>
<td>0.49</td>
<td>0.51</td>
<td>0.51</td>
</tr>
<tr>
<td colspan="6">Physics-aware Video Generation Model</td>
</tr>
<tr>
<td>WISA [15]</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.43</td>
</tr>
<tr>
<td>DiffPhy [8]</td>
<td>0.53</td>
<td>0.59</td>
<td>0.58</td>
<td>0.46</td>
<td>0.54</td>
</tr>
<tr>
<td>CogVideoX-5B [38]</td>
<td>0.39</td>
<td>0.55</td>
<td>0.40</td>
<td>0.42</td>
<td>0.45</td>
</tr>
<tr>
<td>+ PhyT2V [6]</td>
<td>0.45</td>
<td>0.55</td>
<td>0.43</td>
<td>0.53</td>
<td>0.50</td>
</tr>
<tr>
<td>+ SGD [7]</td>
<td>0.49</td>
<td>0.58</td>
<td>0.42</td>
<td>0.48</td>
<td>0.49</td>
</tr>
<tr>
<td>+ PhysHPO [16]</td>
<td>0.55</td>
<td>0.68</td>
<td>0.50</td>
<td>0.65</td>
<td>0.61</td>
</tr>
<tr>
<td>+ Ours</td>
<td>0.67</td>
<td>0.72</td>
<td>0.65</td>
<td>0.60</td>
<td>0.66</td>
</tr>
</tbody>
</table>

The following are the results from Table 2 of the original paper, showing Phenomena Detection (PD) and Physical Order (PO) scores on PhyGenBench:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Mechanics</th>
<th colspan="2">Optics</th>
<th colspan="2">Thermal</th>
<th colspan="2">Material</th>
</tr>
<tr>
<th>PD</th>
<th>PO</th>
<th>PD</th>
<th>PO</th>
<th>PD</th>
<th>PO</th>
<th>PD</th>
<th>PO</th>
</tr>
</thead>
<tbody>
<tr>
<td>Kling [45]</td>
<td>0.61</td>
<td>0.45</td>
<td>0.75</td>
<td>0.55</td>
<td>0.61</td>
<td>0.37</td>
<td>0.62</td>
<td>0.34</td>
</tr>
<tr>
<td>Wan [44]</td>
<td>0.61</td>
<td>0.40</td>
<td>0.76</td>
<td>0.56</td>
<td>0.50</td>
<td>0.27</td>
<td>0.58</td>
<td>0.25</td>
</tr>
<tr>
<td>DiffPhy [8]</td>
<td>0.73</td>
<td>0.53</td>
<td>0.83</td>
<td>0.66</td>
<td>0.70</td>
<td>0.58</td>
<td>0.73</td>
<td>0.43</td>
</tr>
<tr>
<td>Ours</td>
<td>0.79</td>
<td>0.79</td>
<td>0.84</td>
<td>0.85</td>
<td>0.78</td>
<td>0.69</td>
<td>0.75</td>
<td>0.58</td>
</tr>
</tbody>
</table>

The following are the results from Table 3 of the original paper, showing performance on VideoPhy:

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th colspan="3">Overall (%)</th>
<th colspan="3">Solid-Solid (%)</th>
<th colspan="3">Solid-Fluid (%)</th>
<th colspan="3">Fluid-Fluid (%)</th>
</tr>
<tr>
<th>SA, PC</th>
<th>SA</th>
<th>PC</th>
<th>SA,PC</th>
<th>SA</th>
<th>PC</th>
<th>SA,PC</th>
<th>SA</th>
<th>PC</th>
<th>SA, PC</th>
<th>SA</th>
<th>PC</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="13">Video Foundation Model</td>
</tr>
<tr>
<td>VideoCrafter2 [41]</td>
<td>19.0</td>
<td>48.5</td>
<td>34.6</td>
<td>4.9</td>
<td>31.5</td>
<td>23.8</td>
<td>27.4</td>
<td>57.5</td>
<td>41.8</td>
<td>32.7</td>
<td>69.1</td>
<td>43.6</td>
</tr>
<tr>
<td>LaVIE [40]</td>
<td>15.7</td>
<td>48.7</td>
<td>28.0</td>
<td>8.5</td>
<td>37.3</td>
<td>19.0</td>
<td>15.8</td>
<td>52.1</td>
<td>30.8</td>
<td>34.5</td>
<td>69.1</td>
<td>43.6</td>
</tr>
<tr>
<td>SVD-T2I2V [48]</td>
<td>11.9</td>
<td>42.4</td>
<td>30.8</td>
<td>4.2</td>
<td>25.9</td>
<td>27.3</td>
<td>17.1</td>
<td>52.7</td>
<td>32.9</td>
<td>18.2</td>
<td>58.2</td>
<td>34.5</td>
</tr>
<tr>
<td>OpenSora [42]</td>
<td>4.9</td>
<td>18.0</td>
<td>23.5</td>
<td>1.4</td>
<td>7.7</td>
<td>23.8</td>
<td>7.5</td>
<td>30.1</td>
<td>21.9</td>
<td>7.3</td>
<td>12.7</td>
<td>27.3</td>
</tr>
<tr>
<td>Pika [46]</td>
<td>19.7</td>
<td>41.1</td>
<td>36.5</td>
<td>13.6</td>
<td>24.8</td>
<td>36.8</td>
<td>16.3</td>
<td>46.5</td>
<td>27.9</td>
<td>44.0</td>
<td>68.0</td>
<td>58.0</td>
</tr>
<tr>
<td>Dream Machine [49]</td>
<td>13.6</td>
<td>61.9</td>
<td>21.8</td>
<td>12.6</td>
<td>50.0</td>
<td>24.3</td>
<td>16.6</td>
<td>68.1</td>
<td>23.6</td>
<td>9.0</td>
<td>76.3</td>
<td>11.0</td>
</tr>
<tr>
<td>Lumiere [50]</td>
<td>9.0</td>
<td>38.4</td>
<td>27.9</td>
<td>8.4</td>
<td>26.6</td>
<td>27.3</td>
<td>9.6</td>
<td>47.3</td>
<td>26.0</td>
<td>9.1</td>
<td>45.5</td>
<td>34.5</td>
</tr>
<tr>
<td colspan="13">Physics-aware Video Generation Model</td>
</tr>
<tr>
<td>CogVideoX-5B [38]</td>
<td>39.6</td>
<td>63.3</td>
<td>53</td>
<td>24.4</td>
<td>50.3</td>
<td>43.3</td>
<td>53.1</td>
<td>76.5</td>
<td>59.3</td>
<td>43.6</td>
<td>61.8</td>
<td>61.8</td>
</tr>
<tr>
<td>+ PhyT2V [6]</td>
<td>40.1</td>
<td>-</td>
<td>-</td>
<td>25.4</td>
<td>-</td>
<td>-</td>
<td>48.6</td>
<td>-</td>
<td>-</td>
<td>55.4</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>+ Vanilla DPO [51]</td>
<td>41.3</td>
<td>-</td>
<td>-</td>
<td>28.2</td>
<td>-</td>
<td>-</td>
<td>50.0</td>
<td>-</td>
<td>-</td>
<td>51.8</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>+ Ours</td>
<td>49.3</td>
<td>79.5</td>
<td>59.4</td>
<td>40.6</td>
<td>73.4</td>
<td>53.8</td>
<td>60.0</td>
<td>85.6</td>
<td>66.7</td>
<td>54.5</td>
<td>85.4</td>
<td>61.8</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
### Ablation Studies
The authors conduct ablation studies on PhyGenBench to verify the effectiveness of each component of the framework. The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Variant</th>
<th colspan="4">Physical domains (↑)</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>Mechanics</th>
<th>Optics</th>
<th>Thermal</th>
<th>Material</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ours</td>
<td>0.67</td>
<td>0.72</td>
<td>0.65</td>
<td>0.60</td>
<td>0.66</td>
</tr>
<tr>
<td colspan="6">Ablations of PECR module</td>
</tr>
<tr>
<td>w/o PFG</td>
<td>0.63</td>
<td>0.69</td>
<td>0.61</td>
<td>0.53</td>
<td>0.62</td>
</tr>
<tr>
<td>w/o PPD</td>
<td>0.58</td>
<td>0.67</td>
<td>0.61</td>
<td>0.52</td>
<td>0.59</td>
</tr>
<tr>
<td colspan="6">Ablations of TCP module</td>
</tr>
<tr>
<td>w/o PNR</td>
<td>0.65</td>
<td>0.70</td>
<td>0.64</td>
<td>0.56</td>
<td>0.64</td>
</tr>
<tr>
<td>w/o IKS</td>
<td>0.50</td>
<td>0.64</td>
<td>0.58</td>
<td>0.48</td>
<td>0.55</td>
</tr>
</tbody>
</table>

The ablation results show:
1.  Removing the Physics Formula Grounding (PFG) component reduces average performance by 0.04 (6% relative drop), proving that quantitative physical formula constraints are critical for eliminating causal ambiguity.
2.  Removing the Physical Phenomena Decomposition (PPD) component reduces average performance by 0.07 (11% relative drop), proving that decomposing phenomena into ordered event sequences is essential for modeling temporal progression.
3.  Removing the Progressive Narrative Revision (PNR) component reduces average performance by 0.02 (3% relative drop), proving that incremental narrative revision improves semantic coherence across events.
4.  Removing the Interactive Keyframe Synthesis (IKS) component reduces average performance by 0.11 (17% relative drop), proving that explicit visual keyframe priors are the most critical component for ensuring physical consistency and smooth transitions.
### Parameter Analysis
The authors analyze the effect of the number of decomposed events on performance, as shown in Figure 6 from the original paper:

![Figure 6. Effect of event number on Physical Commonsense Alignment (PCA) Score \[9\] across four physical domains: Mechanics, Optics, Thermal, and Material.](images/6.jpg)
*该图像是一个图表，展示了事件数量对四个物理领域（力学、光学、热学和材料学）的物理常识对齐（PCA）得分的影响。随着事件数量的增加，PCA得分在第四个事件数量时达到最高值。*

The results show that performance peaks at 4 events for all physical domains:
- Too few events (1-3) provide insufficient temporal supervision, so the diffusion model cannot accurately follow the physical progression
- Too many events (5-6) introduce accumulated errors in the interactive keyframe editing process, leading to degraded visual quality and physical consistency
  The authors set the default number of events to 4 to balance temporal guidance and keyframe stability.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper proposes a novel event-centric framework for physically plausible video generation, which addresses the core limitation of existing models: their failure to model the causal temporal progression of physical phenomena. The framework consists of two key modules:
1.  The PECR module decomposes physical phenomena into ordered event units, using explicit physical formula constraints to eliminate causal ambiguity in reasoning.
2.  The TCP module converts event units into temporally aligned cross-modal prompts, ensuring smooth transitions between events via progressive narrative revision and interactive keyframe synthesis.
    Comprehensive experiments on two standard PPVG benchmarks show that the framework outperforms all existing state-of-the-art methods by a significant margin (8.19% improvement on PhyGenBench, 3.4% improvement on VideoPhy). Ablation studies confirm the effectiveness of each component, with interactive keyframe synthesis being the most critical for performance.
This work represents a major step forward in PPVG, demonstrating that explicit causal event modeling combined with physical law constraints can significantly improve the physical plausibility of generated videos.
## 7.2. Limitations & Future Work
The authors explicitly point out one key limitation of the framework: it occasionally fails on scenarios governed by multiple overlapping physical laws (compositional physical phenomena), as off-the-shelf foundation models have weak capabilities in compositional physical reasoning. An example failure case is shown in Figure 7 from the original paper:

![Figure 7. Failure case. As foundation models lack capability in compositional physical reasoning, our framework fails to generate scenarios governed by multiple physical laws.](images/7.jpg)
*该图像是一个示意图，展示了牛顿摆的一颗金属球撞击一个充水气球的过程。气球在冲击下爆裂，水喷洒出来，同时其他摆球在重力作用下继续运动。其中涉及的物理法则包括牛顿运动定律、能量守恒及纳维-斯托克斯方程。*

This scenario involves three simultaneous physical laws (Newton's laws of motion, conservation of energy, Navier-Stokes equations for fluid flow), which the framework cannot handle correctly.
The authors propose one primary future research direction: leverage recent advances in compositional visual reasoning to enhance the framework's ability to handle multi-physics scenarios, improving consistency for complex physical phenomena governed by multiple overlapping laws.
## 7.3. Personal Insights & Critique
### Key Inspirations
This work introduces a paradigm shift in PPVG: moving from static prompt augmentation to dynamic event-centric causal modeling. This paradigm is highly generalizable and can be extended to other generative tasks beyond video generation, including:
- 3D scene generation for virtual reality and digital twins
- Physical simulation for robot manipulation training
- Scientific visualization for physics and chemistry research
  The integration of explicit physical formula constraints into the reasoning process is a particularly promising direction, as it bridges the gap between qualitative LLM reasoning and quantitative physical simulation, combining the open-domain flexibility of LLMs with the accuracy of physics engines.
### Potential Improvements
The framework has several areas that can be improved in future work:
1.  **Compositional physical reasoning:** Instead of relying on off-the-shelf foundation models, the framework can be augmented with a dedicated compositional physical reasoning module that can decompose complex scenarios with multiple overlapping physical laws into separate reasoning paths for each law.
2.  **Differentiable physics integration:** The current framework uses physical formulas only for event decomposition; integrating a lightweight differentiable physics simulator into the diffusion sampling process can further improve the physical accuracy of generated videos, especially for complex fluid and rigid body dynamics.
3.  **Longer video generation:** The current framework is optimized for short videos (161 frames); extending the event chain model to handle longer videos with dozens of events would enable the generation of extended physical processes (e.g., a complete chemical reaction, a full building construction sequence).
4.  **Reduction of editing errors:** The current interactive keyframe editing process accumulates errors as the number of events increases; using a consistent identity-preserving image editing model instead of general editing models can reduce these accumulated errors, enabling the use of more events for more fine-grained temporal guidance.
### Unverified Assumptions
One unverified assumption in the paper is that the physical formula retrieval process is always accurate for open-domain prompts. For very niche or complex physical phenomena (e.g., quantum physics effects, non-Newtonian fluid dynamics), the LLM may incorrectly identify the relevant physical law or retrieve the wrong formula, leading to physically incorrect event decomposition. Future work should include a verification step for retrieved formulas, using a dedicated physics knowledge base with human-curated content to reduce retrieval errors.