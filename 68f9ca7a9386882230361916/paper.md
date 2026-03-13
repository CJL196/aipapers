# DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge

Wenyao Zhang124\* Hongsi Liu27\* Zekun Qi34\* Yunnan Wang12\* Xinqiang Yu4 Jiazhao Zhang45 Runpei Dong6 Jiawei He4 Fan Lu7 He Wang45 Zhizheng Zhang4 Li Yi3 Wenjun Zeng2 Xin Jin2‡ 1SJTU 2EIT 3THU 4Galbot 5PKU 6UIUC 7USTC AProject Page Q Code Hugging Face

# Abstract

Recent advances in vision-language-action (VLA) models have shown promise in integrating image generation with action prediction to improve generalization and reasoning in robot manipulation. However, existing methods are limited to challenging image-based forecasting, which suffers from redundant information and lacks comprehensive and critical world knowledge, including dynamic, spatial and semantic information. To address these limitations, we propose DreamVLA, a novel VLA framework that integrates comprehensive world knowledge forecasting to enable inverse dynamics modeling, thereby establishing a perceptionprediction-action loop for manipulation tasks. Specifically, DreamVLA introduces a dynamic-region-guided world knowledge prediction, integrated with the spatial and semantic cues, which provide compact yet comprehensive representations for action planning. This design aligns with how humans interact with the world by first forming abstract multimodal reasoning chains before acting. To mitigate interference among the dynamic, spatial and semantic information during training, we adopt a block-wise structured attention mechanism that masks their mutual attention, preventing information leakage and keeping each representation clean and disentangled. Moreover, to model the conditional distribution over future actions, we employ a diffusion-based transformer that disentangles action representations from shared latent features. Extensive experiments on both real-world and simulation environments demonstrate that DreamVLA achieves $7 6 . 7 \%$ success rate on real robot tasks and 4.44 average length on the CALVIN ABC-D benchmarks.

# 1 Introduction

The evolution of robot learning has demonstrated impressive progress [111] in training policies capable of performing diverse tasks across various environments [1225]. One promising direction is Vision-Language-Action (VLA) models, which leverage the rich understanding capabilities of pre-trained Multimodal Large Language Models (MMLMs) [2629] to directly map natural language instructions and visual observations to robot actions [15, 1, 12]. Although these approaches [30 32, 13, 1, 3342] have achieved impressive results, their direct mapping from observations to actions lacks the closed-loop forecasting capability that humans typically possess when understanding and reasoning about future knowledge of environments.

To incorporate future knowledge prediction into VLA, most existing methods [43, 5, 4455] leverage a copilot generation model to generate future frames/keypoints, then predict action sequences conditioned on goal images. Several methods [5661] integrate pixel-level image forecasting with the action prediction in a single framework, which exploits the synergy of prediction and planning and regards the prediction as an intermediate reasoning step [58] akin to those used in large language models (LLMs) [62]. Despite early success in incorporating dense visual forecasting, these methods naturally exhibit limitations: (1) Redundant pixel information: There exists significant overlap between forecasted images and current observations, making the prediction less efficient and effective. (2) Lack of spatial information: Absence of explicit 3D knowledge of environments [6366, 22]. (3) Lack of high-level knowledge forecasting: Missing high-level understanding of future states, e.g., semantics information. Therefore, we argue that existing methods (Figure 1 (a-c)) are insufficient to forecast future states for a more comprehensive prediction-action loop in the context of world-level future knowledge.

![](images/1.jpg)  

Figure 1: (a) Vanilla VLA directly maps visual observations and language instructions to actions. (b) Models leveraging separate image/video generation or copilot models to generate future frames or trajectories, subsequently guiding an action head. (c) VLA variants explicitly predict a subgoal image as an intermediate visual reasoning step prior to action generation. (d) Our proposed DreamVLA, which explicitly predicts dynamic regions, depth map, semantics (DINOv2 and SAM) knowledge, significantly enhances the model's action reasoning and generalization.

To address these issues, we propose DreamVLA, a novel framework that incorporates comprehensive world knowledge forecasting into the vision-language-action models, thereby establishing a perception-prediction-action loop for the manipulation task. As shown in Figure 1 (d), instead of directly generating entire future frames, our proposed method introduces world embedding to predict comprehensive world knowledge, which is highly relevant to robot execution, such as dynamic area, depth, and high-level semantic features. This approach aligns with the way humans interact with the world, emphasizing relevant changes and world knowledge. By dreaming/forecasting these targeted aspects of the environment, we aim to provide the model with concise and relevant intermediate representations that facilitate more effective action planning.

To obtain comprehensive world knowledge, our approach incorporates three key features: (1) Dynamic region-based forecasting. We leverage an off-the-shelf optical flow prediction model [67, 68] to identify dynamic regions within the scene, enabling the model to concentrate on areas of motion that are critical for task execution instead of redundant frame reconstruction. (2) Depth-aware forecasting. We employ depth estimation techniques [63] to generate per-frame depth maps, providing valuable spatial context that aids in understanding the three-dimensional structure of the environment. (3) High-level foundation features. We incorporate semantic features aligned with visual foundation models such as DINOv2 [69] and SAM [70]. In this way, DreamVLA offers a more comprehensive and effective pathway for the model to plan and execute. Furthermore, we adopt a block-wise structured attention mechanism that masks their mutual attention, preventing information leakage and keeping each representation clean and disentangled. Since the world and action embeddings occupy the same latent space and share similar statistics, a naive MLP head cannot disentangle modality-specific information or exploit their cross-modal correlations. We employ a diffusion-based transformer that disentangles action representations from shared latent features to reason actions.

Through extensive experiments on public benchmarks, we find that incorporating world knowledge prediction leads to significant performance improvements. Our method achieves state-of-the-art performance on the CALVIN benchmark (4.44 average length), and we analyze the influence of the ingredients of our world knowledge and find that they have improvements in different aspects. Specifically, comprehensive ablation shows that predicting dynamic regions alone delivers the greatest gains, while depth and semantic cues offer smaller, roughly equal benefits. Worse, when depth or semantic prediction is used in isolation, it not only fails to help but can actually degrade performance. Extensive experiments on both simulation and real-world demonstrate the effectiveness of our method.

The key contributions of our work are summarized as follows:

•We recast the vision-language—action model as a perceptionprediction—action model and make the model explicitly predict a compact set of dynamic, spatial and high-level semantic information, supplying concise yet comprehensive look-ahead cues for planning.   
We introduce a block-wise structured-attention mechanism, coupled with a diffusion-transformer decoder, to suppress representation noise from cross-type knowledge leakage and thus enable coherent multi-step action reasoning.   

•DreamVLA sets a new state of the art on the CALVIN ABC-D benchmark (4.44 average task length), outperforming prior methods by up to $3 . 5 \%$ on the simulation platform, and boosts real-world success to $7 6 . 7 \%$ . Ablation studies confirm each component's contribution.

# 2 Related Works

# 2.1 Vision-Language-Action Models

The earliest VLA [16, 71, 2, 7274] lay the foundation by combining pretrained vision-language representations with task-conditioned policies for manipulation and control. Inspired by the recent advances of Large Language Models [7578] and multimodal large language models [28, 26, 79, 65, 80] and the emergence of large-scale robot datasets [12, 8183], VLA has become a trend in robot learning. RT series [2, 84, 85] is the pioneer attempt to fine-tune the MLLM on robot demonstration datasets, resulting in strong accuracy and generalization. Building on this foundation, many advanced techniques [30, 32, 13, 1, 33, 34, 73, 3537, 8688, 38, 89] are developed to boost the performance. Meanwhile, considering the advantage of the diffusion model in modeling multi-peak, some researchers [9094] employ different architectures to sample action from noise conditioned on observation, task instruction, and robot prior knowledge. Given on this manner which directly maps observation and instruction to action lacks reasoning steps like LLM [62], most existing methods [43, 5, 4449] leverage a copilot image/video generation model to generate future frames then predict action sequences conditioned on goal images. However, the above methods stll need an extra generation model, which introduces inference time and computation load. Therefore, several methods [5661] integrate pixel-level forecasting with the action prediction in a single framework, which exploits the synergy of prediction and planning. Despite success, these methods naturally exhibit limitations in redundant reconstruction [95], and lack spatial and semantic information.

# 2.2 Knowledge Forecasting for Robotics

Learning future world knowledge for robot training has increasingly become popular to enable policies for achieving an action-forecasting loop. Early attempts [49, 19, 14, 43, 51, 50, 96] to implement this based on off-the-shelf video generation models [97, 53] and feed the goal images or states into policy model to conduct inverse dynamics. This two-stage training strategy is easy to implement but is limited by the performance and latency of video generation models. More advanced solutions couple forecasting with control by requiring the policy to produce, in addition to actions, explicit predictions. Concretely, these works ask the policy to output (i) high-level subtask/option sequences or language plans that decompose long-horizon goals [98100], ii)latent future embeddings/latent actions that compactly encode forthcoming motor intentions [88], (ii)whole sub-goal images or short visual rollouts that anticipate how the scene should evolve [56, 58], and (iv) object-centric signals (e.g., bounding boxes) that capture manipulation-relevant dynamics [83, 87]. This line of work demonstrates better performance and generalization. However, the future states are limited to redundant visual information [63, 64, 101, 69, 102, 66] or monotonous states [21, 48]. In contrast to previous work, DreamVLA proposes to predict future knowledge in an efficient (dynamic region) and effective (comprehensive knowledge) way, demonstrating strong performance and generalization.

# 3 Methodology

# 3.1 Problem Definition and Notation

We aim to improve robot execution by leveraging rich world knowledge as a guiding principle. In this context, we formulate visionlanguage—action reasoning as an inverse dynamics problem [103, 56, 49], which regards the future world knowledge prediction as the intermediate reasoning for robot control, fully unleashing the synergy of prediction and execution. At each time step $t$ , the robot receives three heterogeneous signals: a natural language instruction $l$ , a raw visual frame $o _ { t }$ , and its proprioceptive state $s _ { t }$ . To inject look-ahead reasoning, we define a set of special tokens called <dream> queries [79], and concatenate all inputs into a sequence. A unified model $\mathcal { M }$ maps these inputs into a compact latent representation, which we call the world embedding:

![](images/2.jpg)  

Figure 2: Framework Overview. Given the current robot state $s _ { t }$ , observation $o _ { t }$ , and language instruction, DreamVLA encodes multimodal inputs via frozen text, visual encoders and a tunable state encoder. These tokens, together with a learnable set of <dream> queries, are processed by a large language model to produce world embedding. Three lightweight decoders then project each corresponding element of this embedding into the dynamics region $\hat { f } _ { t + n }$ , monocular depth $\hat { d } _ { t + n }$ and high-level semantics $\hat { c } _ { t + n }$ . A separate <action> query draws a latent action embedding, which conditions a diffusion transformer that refines Gaussian noise into an $n$ -step action sequence $\hat { a } _ { t : t + n - 1 }$ . The dashed box highlights prediction heads that are used only during training; inference skips these heads and operates directly on the world embedding.

$$
\mathbf { w } _ { t + n } = \mathcal { M } \left( l , o _ { t } , s _ { t } \middle | < \middle \mathrm { d } \mathbf { r } \mathbf { e } \mathbf { a m } > \right) .
$$

Next, the world embedding predicts the comprehensive world knowledge that combines motion cues, spatial details and high-level semantics. Specifically, a set of predictor $\mathcal { P }$ extrapolates $n$ steps ahead,

$$
\begin{array} { r } { \hat { p } _ { t + n } = \mathcal { P } \big ( \mathbf { w } _ { t + n } \big ) = \big [ \hat { f } _ { t + n } , \hat { d } _ { t + n } , \hat { c } _ { t + n } \big ] , } \end{array}
$$

where $\hat { f } _ { t + n }$ marks dynamic regions, $\hat { d } _ { t + n }$ encodes monocular depth, and $\hat { c } _ { t + n }$ optionally stores high-level semantic feature (e.g. DINOv2 [69], SAM [70]).

Given world embedding $\mathbf { w } _ { t + n }$ , the <action> query is assigned to the latent action embedding by the unified model $\mathcal { M }$ to aggregate the correlated action information. A denoising-diffusion transformer $\mathcal { D }$ formulates an $n$ -step action based on the latent feature:

$$
\hat { a } _ { t : t + n - 1 } = \mathcal { D } \big ( \mathcal { M } \big ( l , o _ { t } , s _ { t } , < \mathtt { d r e a m } > \vert < \mathtt { a c t i o n } > \big ) \big ) ,
$$

thus completing a perceptionpredictionaction loop that is identical during training and inference. The remainder of this chapter details the system components—encoders, world-knowledge predictor, and diffusion-based action generator—that instantiate the above formulation.

# 3.2 Model Architecture

As illustrated in Figure 2, our DreamVLA framework comprises three core modules operating within a unified transformer architecture. Firstly, heterogeneous inputs—including natural language $l$ visual observations $o _ { t }$ , and proprioceptive states $s _ { t }$ —are individually processed by modality-specific encoders. We encode language instructions using CLIP [101] text embeddings, visual frames through a Masked Autoencoder [104] to obtain spatiotemporal patch representations, and proprioceptive signals via several convolutional and fully-connected layers. Following encoding, a set of learnable queries designated as <dream> and <action> are appended to these multimodal embeddings, where <dream> contains three subqueries (dynamic, depth and semantics), which could be used for the prediction of specific knowledge. Subsequently, we leverage a large language model based on GPT-2 [105] to integrate and attend across modalities and queries using carefully structured causal and non-causal attention mechanisms (Figure 4). This effectively fuses low-level perceptual signals into compact, semantically coherent representations of the world state.

![](images/3.jpg)  

Figure 3: Visualization of dynamic regions over time. We show the static camera (left) and wrist-mounted camera (right) observations alongside the corresponding dynamic masks generated by our method at multiple time steps. The masks highlight dynamic regions by leveraging optical flow trajectories extracted via CoTracker [68, 67]. Compared to the original observations, our method objects and end-effector), enabling more structured and efficient action reasoning.

Finally, specialized light-weight output heads comprising by shallow convolutional layers decode world embedding into explicit predictions: reconstruct anticipated dynamic region, monocular depth, and semantic features. During inference, DreamVLA skips the decoder entirely, saving substantial computation. Instead, the model outputs an world embedding that encapsulates predictions of future dynamics, depth, and semantics without pixel-level reconstruction, thereby retaining the accuracy gains from future-state reasoning while maintaining low latency. In parallel, we employ a denoising diffusion transformer [90] to decode latent action embedding into executable robot action sequences. Collectively, these components enable DreamVLA to perform robust, predictive visionlanguage—action reasoning in an end-to-end manner.

# 3.3 Comprehensive World Knowledge Prediction

Predicting what will matter next is more valuable than merely reproducing the raw future frame. DreamVLA explicitly forecasts future world knowledge that is most relevant for manipulation, including (i) motioncentric dynamic region, (ii) 3D depth geometry, and (ii) high-level semantics. These complementary signals provide a compact, structured surrogate for raw pixels and supply the policy with look-ahead context for inverse dynamics planning.

Motion-centric dynamic-region reconstruction. Predicting dynamic regions tells the robot what parts of the scene are about to move, allowing the model to capture the statistical link between the current scene, the language instruction, and the actions needed to realize the predicted motion. As shown in Figure 3, DreamVLA neither predicts dense optical flow nor synthesizes an entire future frame. Instead, we first apply CoTracker [67, 68] to extract dynamic regions, namely pixels that move with the robot end-effector or other movable objects, and then train DreamVLA to reconstruct only these regions. Furthermore, generating reconstruction targets with an asymmetrical tokenizer can further enhance performance [104]. From the perspective of discrete variational autoencoder (dVAE) [106109], the overall optimization is to maximize the evidence lower bound (ELBO) [110 112, 66] of the log-likelihood $\mathrm { P } ( \bar { x } _ { i } | \tilde { x } _ { i } )$ .Let $x$ denote the original image, $\tilde { x }$ the masked motion region, and $z$ the reconstruction target. The generative modeling can be described as:

$$
\sum _ { ( z _ { i } , \bar { z } _ { i } ) \in \mathcal { D } } \log \mathrm { P } ( x _ { i } | \tilde { x } _ { i } ) \geq \sum _ { ( x _ { i } , \bar { x } _ { i } ) \in \mathcal { D } } \left( \mathbb { E } _ { z _ { i } \sim \mathrm { Q } _ { \phi } \left( \mathbf { z } \mid x _ { i } \right) } \left[ \log \mathrm { P } _ { \psi } ( x _ { i } | z _ { i } ) \right] - D _ { \mathrm { K L } } \left[ z , \mathrm { P } _ { \theta } ( \mathbf { z } | \hat { z } _ { i } ) \right] \right) ,
$$

where $\mathrm { P } _ { \psi } ( x | z )$ is the tokenizer decoder to recover origin data, $\hat { z } _ { i } = \mathrm { Q } _ { \phi } ( \mathbf { z } | \tilde { x } _ { i } )$ denotes the masked motion region tokens from masked data and $\mathrm { P } _ { \theta } ( z | \hat { z } _ { i } )$ reconstructs masked tokens in an autoencoding

fashion. Here, the $\mathrm { P } _ { \theta } ( z | \hat { z } _ { i } )$ is zero, and the dynamic region prediction loss can be formulated as:

$$
\mathcal { L } _ { \mathrm { d y n } } = \frac { 1 } { | \mathcal { D } | } \sum _ { x _ { i } \in \mathcal { D } } \mathbb { E } _ { z \sim Q _ { \phi } ( z | x _ { i } ) } \Big [ - \log \mathrm { P } _ { \psi } \big ( ( x _ { i } ) _ { \mathcal { M } } \mid z \big ) \Big ] .
$$

Depth prediction. Predicting how the depth field will evolve tells the robot where it should move next, steering it toward free space and away from impending obstacles. If depth sensors are available, we supervise the DreamVLA with ground-truth maps; on low-cost platforms without depth sensing, we instead hallucinate future geometry from a single RGB stream. To do so, we treat Depth-Anything [63, 64] predictions as a self-supervised teacher and train a dedicated depth query to regress the aligned future map $\hat { d } _ { t + n }$ . The objective is a scale-normalized mean-squared error,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d e p t h } } = \frac { 1 } { H W } \displaystyle \sum _ { i , j } \big ( \hat { d } _ { t + n } ^ { ( i , j ) } - \alpha { d } _ { t + n } ^ { ( i , j ) } \big ) ^ { 2 } , } \\ & { \quad \quad \alpha = \frac { \sum _ { i , j } \hat { d } _ { t + n } ^ { ( i , j ) } { d } _ { t + n } ^ { ( i , j ) } } { \sum _ { i , j } { d } _ { t + n } ^ { ( i , j ) } } , } \end{array}
$$

where $\alpha$ removes the global scale ambiguity that monocular methods cannot resolve. In practice, this simple loss is sufficient: the teacher provides metrically plausible depth, and the scale-normalization synthesis and collision checking, while ignoring any arbitrary global scale shift.

Contrastive semantic forecasting. Predicting future semantics teaches the robot which objects or regions will matter for the task, providing a high-level context (for example, object identity and affordances) that guides the selection of goals and grasp choice. To learn these semantics, DreamVLA predicts future DINOv2 [69] and SAM [70] feature $\hat { c } _ { t + n }$ using an InfoNCE loss [113, 66]: the ground-truth feature is the positive sample, whereas spatially shifted features act as negatives. This encourages discriminative anticipation that the model must pick the correct object semantics among plausible but wrong futures:

$$
\mathcal { L } _ { \mathrm { s e m } } = - \log \frac { \exp \left( \hat { c } _ { t + n } ^ { \top } c _ { t + n } / \tau \right) } { \sum _ { k } \exp \left( \hat { c } _ { t + n } ^ { \top } c _ { k } / \tau \right) } ,
$$

where $k$ represents the number of tokens in spatial, and $\tau$ denotes the temperature.

Structured attention for cross-type knowledge disentanglement. To preserve clear cross-type knowledge boundaries, <dream> is decomposed into three sub-queries (dynamic, depth and semantics). If these sub-queries could freely attend to one another, highfrequency flow details would contaminate depth reasoning, and semantic cues might bleed into motion features, producing noisy mixed representations. We therefore mask their mutual attention: each subquery attends only to the shared visual, language, and state tokens, while direct links among the three are disabled, keeping their latent features disentangled and free of cross-talk. As shown in Figure 4, both <dream> and <action> queries also employ causal attention restricted to past context, which preserves temporal causality. This organized pattern mirrors the specialist routing used in Mixture-of-Experts (MoE) networks [114]. By avoiding cross-modal leakage, the structured attention supplies clean future world knowledge for action prediction, improves robustness, and maintains temporal consistency.

![](images/4.jpg)  

Figure 4: Block-wise structured attention.

# 3.4 Inverse Dynamics via Denoising Diffusion Transformer

Given two ordered observations $o _ { t }$ and $o _ { t + 1 }$ , classical inverse dynamics infers the intermediate action $\hat { a } _ { t }$ . We extend this formulation by predicting a full action sequence $\hat { a } _ { t : t + n - 1 }$ conditioned on the current observation $o _ { t }$ and future latent world embeddings ${ \bf w } _ { t + n }$ . Specifically, DreamVLA first aggregates this latent embedding, already enriched with predicted future dynamics, depth, and semantics, into a compact action embedding via a dedicated action query and the model's causal attention. Since the world and action embeddings occupy the same latent space and share similar statistics, a naive MLP head cannot disentangle modality-specific information or exploit their crossmodal correlations. We therefore employ a denoising diffusion transformer (DiT) [90, 115] as the action head. Conditioned on the action embedding, DiT employs iterative self-attention and denoising to fuse perceptual forecasts with control priors and to transform Gaussian noise into an $n$ -step trajectory $a _ { t : t + n - 1 }$ , yielding coherent, diverse, and physically grounded action sequences. The loss of action prediction can be formulated as:

Table 1: CALVIN ABC-D results. We present the average success computed over 1000 rollouts for each task and the average number of completed tasks to solve 5 instructions consecutively (Avg. Len.). DreamVLA shows significant superiority over baselines. The best results are bolded.   

<table><tr><td rowspan="2">Method</td><td colspan="6">Task completed in a row</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>Avg. Len. ↑</td></tr><tr><td>Roboflamingo [30]</td><td>82.4</td><td>61.9</td><td>46.6</td><td>33.1</td><td>23.5</td><td>2.47</td></tr><tr><td>Susie [118]</td><td>87.0</td><td>69.0</td><td>49.0</td><td>38.0</td><td>26.0</td><td>2.69</td></tr><tr><td>GR-1 [14]</td><td>85.4</td><td>71.2</td><td>59.6</td><td>49.7</td><td>40.1</td><td>3.06</td></tr><tr><td>3D Diffusor Actor [93]</td><td>92.2</td><td>78.7</td><td>63.9</td><td>51.2</td><td>41.2</td><td>3.27</td></tr><tr><td>OpenVLA [1]</td><td>91.3</td><td>77.8</td><td>62.0</td><td>52.1</td><td>43.5</td><td>3.27</td></tr><tr><td>RoboDual [119]</td><td>94.4</td><td>82.7</td><td>72.1</td><td>62.4</td><td>54.4</td><td>3.66</td></tr><tr><td>UNIVLA [120]</td><td>95.5</td><td>85.8</td><td>75.4</td><td>66.9</td><td>56.5</td><td>3.80</td></tr><tr><td>Pi0 [32]</td><td>93.8</td><td>85.0</td><td>76.7</td><td>68.1</td><td>59.9</td><td>3.92</td></tr><tr><td>CLOVER [121]</td><td>96.0</td><td>83.5</td><td>70.8</td><td>57.5</td><td>45.4</td><td>3.53</td></tr><tr><td>UP-VLA [57]</td><td>92.8</td><td>86.5</td><td>81.5</td><td>76.9</td><td>69.9</td><td>4.08</td></tr><tr><td>Robovlm [37]</td><td>98.0</td><td>93.6</td><td>85.4</td><td>77.8</td><td>70.4</td><td>4.25</td></tr><tr><td>Seer [56]</td><td>96.3</td><td>91.6</td><td>86.1</td><td>80.3</td><td>74.0</td><td>4.28</td></tr><tr><td>VPP [49]</td><td>95.7</td><td>91.2</td><td>86.3</td><td>81.0</td><td>75.0</td><td>4.29</td></tr><tr><td>DreamVLA</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { { D i T } } } = \mathbb { E } _ { \tau , \varepsilon } \big \| \varepsilon - \varepsilon _ { \theta } \big ( \sqrt { \bar { \alpha } _ { \tau } } a _ { t : t + n - 1 } + \sqrt { 1 - \bar { \alpha } _ { \tau } } \varepsilon , \tau , \mathbf { c } \big ) \big \| _ { 2 } ^ { 2 } , } \end{array}
$$

where $\varepsilon _ { \theta }$ is the DiT denoiser, $\varepsilon \sim \mathcal { N } ( 0 , I )$ , $\bar { \alpha } _ { \tau }$ follows a cosine noise schedule and $\mathbf { c }$ is the latent action embedding obtained from a large language model. Inference is performed by drawing a Gaussian sample and running the learned reverse diffusion, yielding diverse yet physically plausible trajectories that close the perception-prediction—action loop.

# 4 Experiments

# 4.1 Implementation Details

All models are implemented in PyTorch and trained on NVIDIA 8 A800 GPUs. We use an AdamW [116] optimizer with initial learning rate $1 0 ^ { - 3 }$ , weight decay $1 e - 4$ , and a cosine learningrate schedule with $5 \%$ linear warm-up. Batch size is set to 64, we set the query length of each modality 9 and diffusion steps in DiT to 10. We weight the dynamic region, depth and segmentation prediction losses as $\lambda _ { \mathrm { d y n } } { = } 0 . 1$ , $\lambda _ { \mathrm { d e p t h } } { = } 0 . 0 0 1$ , $\lambda _ { \mathrm { s e m } } { = } 0 . 1$ , and the action loss as $\lambda _ { \mathrm { D i T } } { = } 1$ , respectively. We first pre-train DreamVLA on the language-free split of the CALVIN [117] and on the full DROID dataset [82]. For the LIBERO benchmark, we first pretrain DreamVLA on LIBERO-90 and then finetune on each track. The model predicts entire frames instead of comprehensive knowledge, keeping storage and computation requirements manageable. We then fine-tune DreamVLA on each target dataset using the comprehensive world knowledge forecasting objective. All models are trained for 20 epochs, and we select the checkpoint with the highest validation success rate (SR) for final evaluation.

# 4.2 Simulation Benchmark Experiments

Simulation setup. We evaluate DreamVLA on CALVIN [117] and LIBERO [122] benchmark. CALVIN is a simulated benchmark designed for learning long-horizon, language-conditioned robot manipulation policies. It comprises four distinct manipulation environments and over six hours of teleoperated play data per environment, captured from multiple sensors including static and gripper-mounted RGB-D cameras, tactile images, and proprioceptive readings. We report the success rate of every track and the average length of 5 tasks. Additionally, evaluations are also conducted on LIBERO [122], a simulated benchmark spanning four suites (LIBERO-Spatial/-Object/-Goal/-Long). Each suite contains 10 tasks supported by 50 human-teleoperated demonstrations, targeting spatial reasoning, object-centric manipulation, and goal completion.

Table 2: The extended LIBERO experiments. DreamVLA achieves the best or competitive performance across all tracks compared to previous approaches. The best results are bolded.   

<table><tr><td rowspan="2">Methods</td><td colspan="4">Scores (%)</td><td rowspan="2">Average</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td></tr><tr><td>Diffusion Policy [90]</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>72.4</td></tr><tr><td>Octo [13]</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td></tr><tr><td>OpenVLA [1]</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>SpatialVLA [36]</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td></tr><tr><td>CoT-VLA [58]</td><td>81.1</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td></tr><tr><td>DreamVLA</td><td>97.5</td><td>94.0</td><td>89.5</td><td>89.5</td><td>92.6</td></tr></table>

Results. As shown in Table 1, DreamVLA achieves the highest performance on ABC-D tasks, Our method surpasses Roboflamingo [30], 3D Diffusor Actor [93], OpenVLA [1], RoboDual [119], UNIVLA [120], Robovlm [37] and GR1 [14], which directly projects the RGB/depth image to action signals as shown in Fig. 1(a) in the manuscripts. Compared to methods that use a copilot model to generate sub-goal images as input, like Susie [118] and CLOVER [121] as shown in Fig. 1(b) in manuscripts, our model significantly achieves more accurate control. DreamVLA outperforms approaches like UP-VLA [57], Seer [56], and VPP [49] as shown in Fig. 1(c) in manuscripts, which merge whole sub-goal image foresight into one VLA to take benefits from a more integrated design and joint optimization. indicating that our method has better multi-task learning and generalization capabilities in simulation tasks. For the LIBERO benchmark [122], DreamVLA exhibits better or comparable ability across all tracks compared to previous approaches by future world knowledge prediction as shown in Table 2.

# 4.3 Real World Experiments

To evaluate the effectiveness of our method in the real-world, we use the Franka Panda arm to conduct real-world experiments on gripper grasping. In our setups, two RealSense D415 cameras capture RGB images. One is in a third-person view, and the other is at the end of the robotic arm, as shown in Figure 5. We collect four categories of objects for two tasks: pick and place. Additionally, we conduct experiments on drawer opening and closing tasks, as shown in the supplementary. Follow [56], we pretrain DreamVLA on the DROID [82] contains large-scale trajectories of Franka robots in varied scenes. For fair comparison, we fine-tune Diffusion Policy [90], Octo-Base [13], OpenVLA [1] and DreamVLA on collected demonstration datasets containing 100 trajectories for each task.

![](images/5.jpg)  

Figure 5: Real-world experiment setup.

In the experimental setup, each trial permits a maximum of 20 consecutive attempts. For the grasping experiments, objects are randomly positioned on the table surface. A trial is deemed successful if the robotic arm successfully grasps the target object within the predefined attempt limit. In the placement experiments, the robot is required to heasnt bas  ot e and placement operations are completed within the allowed attempts. For the drawer manipulation tasks, the drawer is placed randomly in front of the robotic arm. The experiment is considered successful if the drawer displacement exceeds 10 centimeters, indicating effective interaction. The results, presented in Table 3, demonstrate that our method performs better than other methods. More real-world experiment visualizations are shown in the supplementary section.

Table 3: Real-world evaluation with the Franka Robot across three tasks.   

<table><tr><td rowspan="2">Method</td><td colspan="3">Pick</td><td colspan="3">Place</td><td colspan="3">Drawer</td><td>Task (All)</td></tr><tr><td>Bottle</td><td>Doll</td><td>Avg.</td><td>Banana</td><td>Chili</td><td>Avg.</td><td>Open</td><td>Close</td><td>Avg.</td><td>Avg.</td></tr><tr><td>Diffusion Policy [90]</td><td>50.0</td><td>70.0</td><td>60.0</td><td>65.0</td><td>45.0</td><td>55.0</td><td>15.0</td><td>60.0</td><td>37.5</td><td>50.8</td></tr><tr><td>Octo-Base [13]</td><td>50.0</td><td>60.00</td><td>55.0</td><td>40.0</td><td>50.0</td><td>45.0</td><td>20.0</td><td>50.0</td><td>35.0</td><td>45.0</td></tr><tr><td>OpenVLA [1]</td><td>50.0</td><td>40.0</td><td>45.0</td><td>20.0</td><td>30.0</td><td>25.0</td><td>40.0</td><td>30.0</td><td>35.0</td><td>35.0</td></tr><tr><td>DreamVLA</td><td>85.0</td><td>80.0</td><td>82.5</td><td>80.0</td><td>80.0</td><td>80.0</td><td>70.0</td><td>65.0</td><td>67.5</td><td>76.7</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="6">Task completed in a row</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>Avg. Len. ↑</td></tr><tr><td>Vanilla VLA*</td><td>93.0</td><td>82.4</td><td>72.3</td><td>62.6</td><td>53.3</td><td>3.64</td></tr><tr><td>+ dynamic region</td><td>97.6</td><td>92.6</td><td>87.5</td><td>80.4</td><td>73.7</td><td>4.32</td></tr><tr><td>+ depth</td><td>98.3</td><td>94.3</td><td>88.5</td><td>82.0</td><td>77.2</td><td>4.40</td></tr><tr><td>+ semantics</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

Table 4: Performance comparison between predicting the optical flow and dynamic region. Notably the \* denotes that this result is from [56].   

# 4.4 Ablation Study

In this section, we design the experiments to investigate the following questions.

# Q1: What is the contribution of each modal characteristic?

The core motivation of DreamVLA is to enable the model to predict comprehensive visual knowledge of the future to enhance action reasoning. However, not all types of knowledge contribute equally to subsequent execution. We consider four types of predictive knowledge: dynamic region, depth, and semantic segmentation features derived from SAM and DINO. As shown in Figure 6, we first train the model with each knowledge forecasting independently. The green dashed line denotes the performance of the Vanilla VLA baseline, which uses no knowledge prediction. Among all, predicting dynamic regions proves to be the most beneficial, because these masks explicitly flag the pixels that are about to change and therefore align almost perfectly with the policy's action semantics. By contrast, supervising the network with depth map, DINO or SAM features alone not only fails to help but often degrades performance. We analyze that this gap stems from how closely each auxiliary target matches the downstream objective: dynamic-region labels supply gradients that reinforce the action head, whereas depth regression and high-dimensional feature matching (DINO/SAM) inject large, noisy losses that dominate optimization. With the limited model attention budget, these competing gradients dilute the task-relevant features and push the backbone toward suboptimal optima, producing the observed drop below the dashed baseline.

Next, we train the model with all five knowledge heads simultaneously (All) and perform an ablation study (All-X), where we remove one knowledge signal at a time to evaluate its contribution. Removing Flea tothe most igificant perormn drop, confirmi s essential rol. Interestingly, remi DINO results in similar or even better performance, suggesting that not all semantic signals are equally helpful or stable in predicting outcomes, so we only use semantic features from SAM in the subsequent ablations. Table 4 reveals a clear and decreasing return pattern in all ablations.

# Q2: Auxiliary Tasks vs. Future Knowledge Prediction: which drives improvement?

Table 5 contrasts two training regimes: predicting complete world knowledge and performing auxiliary reconstructions, showing that the former is decisively superior. In our ablation, every prediction strategy is individually replaced by its reconstruction counterpart, yet each substitution consistently lowers performance: VLA trained only to redraw the current RGB, depth, semantics, or DINOv2 features can handle the first few actions but soon loses coherence, whereas a network trained to forecast the next dynamic region, depth map, and semantics preserves accuracy throughout the trajectory and carries tasks much farther before failure. The reason is that prediction provides a richer, action-oriented signal, directing learning toward the pixels that will drive the upcoming decision, while reconstruction merely revisits background detail that the control policy never actually needs.

Q3: Why do we use the optical flow as the mask instead of directly forecasting it?

![](images/6.jpg)  

Figure 6: CALVIN ABC-D performance with respect to different combinations of knowledge prediction. $\mathbf { A l l = a l l }$ of five models, and All- $\mathbf { \nabla } \cdot \mathbf { X } =$ taking X out of All.

Table 5: Performance comparison between cotraining with auxiliary tasks and predicting the comprehensive world knowledge.   

<table><tr><td rowspan="2">Method</td><td colspan="6">Task completed in a row</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>Avg. Len.</td></tr><tr><td>Auxiliary</td><td>97.7</td><td>92.3</td><td>85.6</td><td>79.5</td><td>74.2</td><td>4.14</td></tr><tr><td>Prediction</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="6">Task completed in a row</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>Avg. Len.</td></tr><tr><td>Optical</td><td>97.6</td><td>92.4</td><td>86.8</td><td>81.7</td><td>75.4</td><td>4.23</td></tr><tr><td>Dynamic</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

Table 6: Performance comparison between predicting the optical flow and dynamic region.   

To justify our choice of employing motion-centric dynamic regions over direct flow forecasting, we implement both variants under identical settings (Table 6). In the optical flow setup, the model must predict the full future flow field along with the subgoal image, which significantly increases the training complexity. This extra burden manifests in markedly lower multi-step success rates. By contrast, our dynamic region approach merely employs the pretrained fow model to obtain a binary mask, focusing the model on "where" relevant motion occurs, bringing a significant improvement.

# Q4: The effectiveness of structured attention in DreamVLA.

To demonstrate the effectiveness of our proposed structure attention mechanism in Figure 4, we swap it for a vanilla causal mask while keeping everything else fixed. In this setting, every <dream> query, including the one meant to capture semantics, can also read the flow and depth tokens produced in the same step; the extra cross-peek mixes unrelated signals, adds gradient noise, and quickly degrades long-horizon control. Our mask removes all query-to-query edges, so <action> query consults only past language, state and multimodal predictions, never their siblings. Table 7 shows the payoff: the causal variant brings a marginal improvement for Vanilla VLA, whereas the block-sparse version keeps success high throughout, confirming that blocking intra-step leakage is important.

# Q5: Can we use the shared query to predict the comprehensive world knowledge?

Instead of assigning separate queries to dynamic region, depth, and semantics features, one might let a single set of shared queries predict all signals. To test this idea, we split each world-embedding vector into four equal sub-spaces, with each quarter intended to carry a different modality. Table 8 shows that the shared-query design hurts action performance: mixing modalities in the same query introduces cross-talk, so the diffusion head receives noisy features. In contrast, giving each modality its query keeps the representations disentangled and yields a clear performance gain.

# Q6: Effect of the query count per modality inside <dream> queries.

Each <dream> query contains three groups of elements: dynamic, depth, and semantics, each assigned $K$ queries. We vary $K \in \{ 4 , \bar { 9 } , 1 6 \}$ to examine its influence. When $K =$ 4, the limited capacity prevents the model from encoding fine-grained motion, geometry, and semantics, so accuracy drops even though memory us

<table><tr><td rowspan="2">Number</td><td colspan="6">Task completed in a row</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td> Avg. Len.</td></tr><tr><td>4</td><td>97.2</td><td>92.6</td><td>86.4</td><td>80.7</td><td>75.1</td><td>4.32</td></tr><tr><td>9</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr><tr><td>16</td><td>98.1</td><td>93.0</td><td>86.9</td><td>81.0</td><td>73.9</td><td>4.33</td></tr></table>

age is lowest. With $K = 9$ , each modality has sufficient bandwidth without overloading the backbone, yielding the best success rate and the longest uninterrupted task execution. Increasing to $K = 1 6$ introduces redundant tokens that compete for attention and raise GPU memory, bringing no extra gain and slightly lower generalization.

Table 9: Performance comparison between different numbers of <dream> queries.   

Table 7: Performance comparison between vanilla causal and our structured attention.   

<table><tr><td rowspan="2">Method</td><td colspan="6">Task completed in a row</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>Avg. Len.</td></tr><tr><td>Causal</td><td>94.2</td><td>86.5</td><td>78.4</td><td>71.3</td><td>62.7</td><td>3.75</td></tr><tr><td>Structure</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="6">Task completed in a row</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>Avg. Len.</td></tr><tr><td>Shared</td><td>95.5</td><td>90.1</td><td>83.8</td><td>76.9</td><td>70.4</td><td>4.17</td></tr><tr><td>Separated</td><td>98.2</td><td>94.6</td><td>89.5</td><td>83.4</td><td>78.1</td><td>4.44</td></tr></table>

Table 8: Performance comparison between shared and seprated queries.   

# 5 Limitation & Future Works

While DreamVLA demonstrates solid vision-language-action and achieves state-of-the-art performance on CALVIN [117], its current scope is still narrow: it practises mainly parallel-gripper manipulation, relies on RGB-centric data, and is trained on scenes with limited geometric and material diversity. We therefore plan to (i) add dexterous-hand demonstrations with rich contact annotations [123, 124], (ii) introduce 3D point clouds [125, 126, 102, 66, 127, 128, 65, 129] and spatial information [22, 130], tactile—and fuse them into volumetric world states, and (iii) extend data collection and on-policy fine-tuning to bolster generalization and long-horizon robustness.

# 6 Conclusion

We present DreamVLA, a novel Visual-Language-Action framework that enables inverse dynamics modeling through comprehensive world knowledge prediction, supporting the perception-predictionaction loop for manipulation tasks. DreamVLA leverages dynamic-region-guided knowledge forecasting, combining spatial and semantic cues to generate compact and informative representations for action planning. We introduce a block-wise structured-attention mechanism, coupled with a diffusion-transformer decoder, to suppress representation noise from cross-type knowledge leakage and thus enable coherent multi-step action reasoning. Extensive experiments in both real and simulated environments demonstrate the effectiveness of DreamVLA, achieving a $7 6 . 7 \%$ success rate on real-world robot tasks and outperforming prior methods on the CALVIN ABC-D benchmark.