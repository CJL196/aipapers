# SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning

Hanzhen Wang1\*, Jiaming $\mathbf { X } \mathbf { u } ^ { 1 , 3 ^ { * } }$ , Jiayi $\mathbf { P a n } ^ { 1 , 2 }$ , Yongkang Zhou1.3, Guohao Dai1,2,3† 1Shanghai Jiao Tong University, 2Infinigence-AI, 3SII \*These authors contributed equally. †Corresponding author: daiguohao@sjtu.edu.cn

# Abstract

Pruning is a typical acceleration method for compute-bound problems by effectively reducing the computation amount. Recently, pruning has been applied to the Vision-LanguageAction (VLA) models and emerges as a promising acceleration method by evicting unimportant tokens. However, existing pruning methods only focus on the local information in current action generation and ignore the global information in previous actions, resulting in a reduction of more than $20 \%$ in the success rate and limited speedup in some scenarios. In this paper, we point out that the information across consecutive actions exhibits a high degree of similarity, and thus propose the novel insight that combines local information in the current action generation with global information from previous generation for token selection. Based on the insight, we further propose SpecPrune-VLA, a training-free pruning method including two-level token pruning with heuristic control. (1) Static token pruning at action level. We explore the token redundancy through global information in previous actions and the token through local information in the current action generation, statically reducing the number of visual tokens at the action level. (2) Dynamic token pruning at layer level. We exploit the relevance between the tokens and model layer, and dynamically prune tokens based on their layer-specific importance at the layer level. (3) Lightweight action-aware controller. We point out that the generated action can be categorized into coarse-grained and fine-grained based on the speed, with fine-grained actions being sensitive to the error brought by pruning. Therefore, we introduce a lightweight controller that can identify the current action granularity and adjust the pruning strategy accordingly. Extensive experiments show that, compared with the high-performing VLA model OpenVLA-OFT, SpecPrune-VLA achieves an average $1 . 4 6 \times$ speedup on NVIDIA A800 GPU and $1 . 5 7 \times$ speedup on NVIDIA GeForce RTX 3090 GPU with negligible loss on task success rate in the LIBERO simulation benchmark.

# 1 Introduction

Vision-Language-Action (VLA) models, built on the large language model (LLM), have attracted much attention for their ability to understand multimodal information and generate robotic action. Previous works like RT-1 (Brohan et al. 2022) and OpenVLA (Kim et al. 2024) exhibit remarkable performance in cross-task generalization and instruction following by learning from real-world robotic interaction. Subsequent works (Team et al. 2024; Li et al. 2024; Black et al.

![](images/1.jpg)  

Figure 1: (a) The mainstream inference dataflow of VLA models. (b) Latency breakdown in three typical VLA models in the LIBERO benchmark during each action generation. (c) The practical arithmetic intensity of three models in the roofline model of NVIDIA A800 GPU.

2024; Kim, Finn, and Liang 2025) further propose more delicate model structures, achieving real-time performance and efficiency improvement. Figure 1(a) shows the mainstream inference dataflow of VLA models, consisting of three components: (1) tokenizers that convert multimodal inputs into tokens (e.g., image and text encoders), (2) LLM backbone that understands the multimodal inputs and generates the intermediate results, and (3) an action head that generates continuous low level actions based on the intermediate results. We further select three representative VLA models, OpenVLA (Kim et al. 2024), CogACT (Li et al. 2024), and OpenVLA-OFT (Kim, Finn, and Liang 2025) to profile the latency breakdown of these three components during each action generation in the LIBERO benchmark shown in Figure 1(b). Statistical data reveal that OpenVLA-OPT is the high-performing model to date in inference efficiency, and the LLM s te cal bottlek, ti o o tha $70 \%$ of the end-to-end inference latency.

![](images/2.jpg)  
ve

Therefore, most works in VLA model acceleration focus on the LLM backbone acceleration and have explored techniques such as quantization (Park et al. 2024), early exit (Yue et al. 2024), and caching (Xu et al. 2025b) to accelerate VLA model inference. However, they ignore the unique computation characteristics of VLA models, resulting in limited performance and effectiveness. Nowadays, the latest VLA models (e.g., CogACT and OpenVLA-OPT) adopt the singlestep inference paradigm that the model directly predicts a sequence of low-level actions through single LLM forward with hundreds of multimodal tokens. As a result, from the perspective of arithmetic intensity (i.e., the amount of computation per byte) in the hardware roofline model, the VLA model inference is primarily compute-bound, shown in Figure 1(c), where latency is mainly determined by the computation amount rather than memory access. Pruning is a typical acceleration method for compute-bound problems by effectively reducing the computation amount. However, existing token-pruning methods in VLA models only consider the local information (e.g., the layer results in current action generation) and ignore the global information across the whole model, leading to either $> 2 0 \%$ success rate loss or limited speedup in some scenarios. In this paper, we observe that input images in consecutive action generations exhibit high similarity due to the short temporal intervals between them. Therefore, we consider that the global information from previous inference steps can be leveraged for more reliable and efficient token pruning. Based on the above insight, we propose SpecPruneVLA, an acceleration method for Vision-Language-Action Models through action-aware self-speculative pruning. The techniques of SpecPrune-VLA can be summarized into three points as follows. (1) Static token pruning at action level. Based on the insight, we point out that the tokens between consecutive action generation are largely overlapped (e.g., the background in environment images), leading to significant redundancy. Therefore, we reuse the attention information from the last generation to prune unimportant tokens and retain a globally highly-weighted token set. Then we enhance it with dynamic elements and task-relevant tokens by speed-based frame comparison and self-speculative token selection. By fusing tokens selected from both local and global levels, we can prune over $50 \%$ to $70 \%$ visual tokens at the beginning of LLM forward. (2) Dynamic token pruning at layer level. As the input features propagate through the LLM backbone, the local context of each token is progressively enriched by deeper layers. Therefore, we introduce layer-wise pruning by dynamically updating tokens' importance scores and reevaluating token importance at different depths. This allows the model to adaptively refine its computation focus, removing redundant tokens as contextual understanding matures. (3) Lightweight action-aware Controller. We propose that not all actions are equally sensitive to token pruning. Therefore, we categorize actions into coarse-grained (e.g., large translations or rotations) and fine-grained (e.g., grasping) types and design a controller that determines action granularity based on the speed of the end-effector and adaptively adjusts the pruning strategy. It enables a robust tradeoff between speed and accuracy across diverse robotic tasks. We implement SpecPrune-VLA on both the NVIDIA A800-80GB and the NVIDIA GeForce RTX 3090 GPUs to evaluate the performance. Compared to the high-performing model OpenVLA-OFT, our method achieves an average $1 . 4 6 \times$ speedup on the A800 and $1 . 5 7 \times$ speedup on the RTX 3090, with negligible degradation in task success rate on the LIBERO simulation benchmark. The consistent speedup gains across both platforms demonstrate that SpecPrune

![](images/3.jpg)  

Figure 3: (a) The original image the model sees; (b) The images where unimportant tokens are pruned; (c) The tokens are randomly pruned, some important tokens are pruned; (d) Important tokens(e.g., the tomato sauce) are pruned. (e) The influence of different pruning strategies and pruning numbers of tokens on performance.

VLA generalizes well to different hardware architectures.

# 2 Related Works

# 2.1 Vision-Language-Action (VLA) Models

VLA models are typically built upon Large Language Model (LLM) structure (Zitkovich et al. 2023; Liu et al. 2023b; Wang et al. 2024), and are further fine-tuned on extensive datasets from both simulated (Liu et al. 2023a) and realworld (O'Neill et al. 2024) robotic environments. The model receives multimodal information (e.g., images and text) and generates low-level robotic actions. It is investigated that a continuous action space has better performance in manipulation accuracy (Liu et al. 2025). Therefore, an action head, such as a lightweight MLP and diffusion model (Liu et al. 2024; Wen et al. 2025) is introduced to convert action hidden states to continuous actions. Besides, to achieve high control frequency and action coherence, modern VLA models leverage ACT (Zhao et al. 2023), diffusion models (Peebles and Xie 2023), or parallel decoding to generate action chunks.

# 2.2 Token-level acceleration for VLA models

Existing works have explored token caching or pruning for VLA models. For example, VLA-Cache (Xu et al. 2025b) caches the similar and unimportant tokens' key and value from the last generation and reuses them in the next inference. However, the effect is limited since it only reduces $50 \%$ to $7 5 \%$ FLOPs in attention matrix calculation, about $17 \%$ to $2 5 \%$ of all FLOPs in LLM. Visual-token pruning methods such as EfficientVLA (Yang et al. 2025) propose to identify task-relevant tokens using attention maps from a single LLM layer and complement retained tokens with visually diverse patches to preserve coverage. However, this approach relies on a single-layer heuristic that may not reflect global contextual information, and the diversity-driven supplementation can inadvertently introduce task-irrelevant content, limiting its reliability across diverse environments. Besides, (Li et al. 2025) aims to preserve both spatial information and semantic content by retaining tokens with high feature saliency from the vision encoder. While this improves structural integrity, it does not explicitly distinguish between semantically rich and task-irrelevant tokens, which still contribute unnecessary computational load.

![](images/4.jpg)  

Figure 4: For the task "turn on the oven and put the bottle on it. (a) Prune the tokens relying on local information(e.g., attention scores from one LLM layer). (b) Prune the tokens relying on the global attention of the LLM in the last action generation. (c) The practically important tokens after inference completion.

# 2.3 Self-speculative decoding and lightweight predictor

Standard speculative decoding (Leviathan, Kalman, and Matias 2023) requires an additional draft model to generate candidate tokens. In contrast, LayerSkip (Elhoushi et al. 2024) proposes a self-speculative method that leverages the early layers of the same model as the draft model, eliminating the need for an external model; the remaining layers are used for verification and correction, thereby reducing memory footprint and improving inference efficiency. Besides, the draft model can also be used as a predictor to provide estimates of helpful information. In SpecEE ( $\mathrm { { X u } }$ et al. 2025a), a small model is utilized as a predictor to narrow down the candidate vocabulary. The predictor filters out unlikely candidates, allowing the model to search within a much smaller vocabulary. This significantly reduces computational overhead.

# 3 Key Insights

# 3.1 What Really Matters in the Image

We first carry out a systematic study on critical components of the images for VLA models. Our focus is on ensuring that the model can effectively recognize task-relevant objects and dynamic elements to accurately perceive the spatial context of the target. Tokens that significantly contribute to this perception are referred to as important tokens.

To validate the effectiveness of our idea, we conduct a simple experiment on some tasks. We let the model complete a generation but don't act in the environment; then we use the output attention score of the last layer to prune unimportant tokens with low scores. After that, we input the pruned images and let the model regenerate the action and act in the environment. The Figure 3(e) shows that pruning a small number of tokens at random has minimal impact on performance, indicating the presence of redundant tokens in the image. However, as more tokens are pruned, the model's performance degrades sharply. It is worth noting that more than half of the unimportant tokens can be pruned without a significant drop in success rate (compared to $9 3 . 3 \%$ in the none pruning case). This experimental result provides strong support for the reliability of the pruning approach. Besides, the model suffers drastic performance degradation (up to $80 \%$ in success rate) when important tokens are pruned. It further highlights that important tokens need to be retained.

# 3.2 Redundancy largely overlaps in images of consecutive inference

Tokens other than important tokens are considered redundant. It is challenging to identify redundant tokens before the model completes the current inference. Current methods, such as (Li et al. 2025; Yang et al. 2025), utilize local information such as the attention score of one LLM layer or the vision encoder. However, they fail to leverage global information from the whole model and are thus not reliable as shown in Figure 4(a) and (c). The important tokens in these two cases vary obviously. In this paper, we emphasize that in VLA models, the overall task goal remains constant throughout the entire execution, and a large proportion of the visual scene remains unchanged across consecutive frames. Therefore, tokens identified as redundant by the model in the previous generation are highly likely to remain redundant in the current step as shown in Figure 4(b) and (c). This temporal consistency inspires us to reuse the global attention across time.

# 4 Static Token Pruning at Action Level

# 4.1 Observation and Insight

Due to changing sub-goals and dynamic visual elements, it is necessary to incorporate information of the current generation. We analyze attention-based importance using Equation (1) and observe that $8 5 \% - 9 5 \%$ of top $\mathbf { \nabla \cdot k }$ tokens in the first two layers reappear in the final layer's top- $\mathbf { \nabla } \cdot \mathbf { k }$ as shown in Figure5, where top- $\mathbf { \nabla } \cdot \mathbf { k }$ equals 20. We define this overlap as the hit rate. The high hit rate indicates that early-layer attention provides a reliable local guidance for important token selection. Besides, the first layer alone exhibits a low hit rate, while incorporating both the second and third layers provides only marginal gains in hit rate and introduces extra latency. Therefore, we choose the first two layers for speculation.

# 4.2 Method

Pruning based on global information Attention score reflects the degree to which other tokens attend to a given token, and we can use attention scores to filter out unimportant visual tokens. The attention score for a visual token to the task is computed as follows: for a given layer, let $V _ { i }$ be a visual token and $T = \{ t _ { 1 } , t _ { 2 } , \dots , t _ { m } \}$ be the set of text sequence describing the task. Then, the task attention score of $V _ { i }$ in layer $l$ is:

![](images/5.jpg)  

Figure 5: (a) Comparison of the hitrate between leveraging the first one, two, and three layers. (b) LLM latency comparison between leveraging the first one, two, and three layers.

![](images/6.jpg)  

Figure 6: The selection strategy for frame comparison.

$$
\mathrm { S c o r e } _ { l } ( V _ { i } ) = \frac { 1 } { H \cdot m } \sum _ { h = 1 } ^ { H } \sum _ { j = 1 } ^ { m } A _ { l } ^ { h } ( V _ { i } , t _ { j } ) ,
$$

where $A _ { l } ^ { h } ( V _ { i } , t _ { j } )$ denotes the attention weight from head $h$ in layer $l$ of the cross-attention (or self-attention) mechanism from $V _ { i }$ to text token $t _ { j }$ , $H$ is the number of attention heads, and $M$ is the number of task-relevant text tokens. We define $V _ { \mathbf { g l o b a l } }$ as the set of the top- $K _ { G }$ most important tokens with the highest global attention score from prior inference. Supplementation of Dynamic Tokens Visual tokens undergoing substantial changes between inference steps cannot be accurately pruned based on global information from the last inference. Therefore, in the current inference step, we explicitly identify and preserve these dynamic tokens to ensure that up-to-date information is retained during token pruning. For two images $I _ { 1 }$ and $I _ { 2 }$ , we first divide the images into $N \times N$ patches according to the visual token size. Then, let $\mathbf { P } _ { t } ^ { i , j }$ denote the feature vector of the patch at position $( i , j )$ in frame $I _ { n }$ . We compute the cosine similarity between corresponding patches as:

$$
\mathrm { S i m } ( \mathbf { P } _ { m } ^ { i , j } , \mathbf { P } _ { n } ^ { i , j } ) = \frac { \mathbf { P } _ { m } ^ { i , j } \cdot \mathbf { P } _ { n } ^ { i , j } } { \| \mathbf { P } _ { m } ^ { i , j } \| _ { 2 } \| \mathbf { P } _ { n } ^ { i , j } \| _ { 2 } } ,
$$

where $\| \cdot \| _ { 2 }$ denotes the Euclidean norm. To identify dynamic tokens, we first filter patches with similarity scores below a threshold $\tau$ , then select the top- $k$ patches with the lowest similarity scores from the remaining candidates. Formally, let $\mathcal { P } _ { n } = \mathbf { \bar { \{ P } }  _ { n } ^ { i , j } \mid 1 \leq i , j \leq N \}$ be the set of all patches in frame $I _ { n }$ . We define the candidate dynamic patches as those with significant changes:

$$
\mathcal { C } _ { n } = \left\{ \mathbf { P } _ { n } ^ { i , j } \in \mathcal { P } _ { n } \left| \mathrm { S i m } ( \mathbf { P } _ { m } ^ { i , j } , \mathbf { P } _ { n } ^ { i , j } ) < \tau \right. \right\} .
$$

![](images/7.jpg)  

Figure 7: Detailed implementation of static token pruning. We prune the tokens based on the global information from the attention scores in the last action generation, the input image comparison, and the local information from the selfspeculative results in the first two layers.

The most dynamic $K _ { D }$ tokens are then given by:

$$
V _ { \mathrm { d y n a m i c } } = \mathrm { L o w } { - } K _ { D } \left( \{ \mathrm { S i m } _ { i , j } \ | \ \mathbf { P } _ { t } ^ { i , j } \in \mathcal { C } _ { t } \} \right) ,
$$

Additionally, as illustrated in Figure 6(a), directly comparing adjacent frames can lead to inaccurate results. To mitigate these disturbances and enhance sensitivity to dynamic tokens, we propose a velocity-based frame sampling strategy. This approach selects a historical reference frame that is T frames prior to the current frame, where $\mathrm { T }$ is determined sed on theavrge moven $\begin{array} { r } { T = \left[ \frac { - 1 6 } { 3 } \cdot \frac { v } { 6 } + \frac { 2 2 } { 3 } \right] + 4 } \end{array}$ n'sead $v _ { t }$ denotes the movement speed of the end effector. The calculation of speed is discussed in Section 6. This adaptive strategy ensures that we can effectively identify dynamic tokens while minimizing the influence of irrelevant variations and maintaining temporal precision. Pruning based on local information Our observation and insights suggest that the initial two layers can serve as a reliable predictor to filter task-relevant tokens for current steps. For the first two decoder layers, we compute the task attention scores for all visual tokens as defined in Equation (1). In each layer, we select the $K _ { b a s e }$ visual tokens with the highe ttenion o    ndi $V _ { ( 1 ) }$ and $V _ { ( 2 ) }$ respectively, and take the union of these two sets as the local information representation:

$$
V _ { \mathrm { l o c a l } } = V _ { ( 1 ) } \cup V _ { ( 2 ) }
$$

Finally, all the pruning token set is:

$$
V _ { \mathrm { p r u n e } } = U - V _ { \mathrm { g l o b a l } } \cup V _ { \mathrm { d y n a m i c } } \cup V _ { \mathrm { l o c a l } }
$$

and $U$ is the set of all tokens.

# 5 Dynamic Token Pruning at Layer Level 5.1 Method

To preserve the most important tokens in layers, we propose an importance scoring mechanism that leverages attention scores and layer confidence across transformer layers to prune tokens within layers. Importance Score Formulation The token importance score is initialized for the remaining visual tokens after static token pruning and subsequently updated in the target transay $s _ { i } ^ { ( l ) }$ takes into account both the relative importance weight of tokens and the layer contribution. Therefore, we compute the importance score $s _ { i } ^ { ( l ) }$ for each token $t _ { i }$ in the remaining visual token set $T$ based on two key components:

$$
s _ { i } ^ { ( l ) } = \omega _ { \mathrm { r a n k } , i } ^ { ( l ) } \times \omega _ { \mathrm { c o n f } } ^ { ( l ) }
$$

where ωrank,i denotes rank-based weight reflecting the token's relative portance in attention ranking, denotes layer confidence score measuring the layer's reliability Rank-based Weight For each attention head, tokens are ranked based on their attention scores. To emphasize the contribution of the most important tokens while maintaining a smooth decay in influence, we introduce a rank-based weighting scheme. This weight is defined as:

$$
\omega _ { \mathrm { r a n k } , i } ^ { ( l ) } = \frac { \sigma ( - k \cdot \mathrm { r a n k } _ { i } ^ { ( l ) } ) } { \sum _ { j } \sigma ( - k \cdot \mathrm { r a n k } _ { j } ^ { ( l ) } ) }
$$

where $\mathrm { r a n k } _ { i } ^ { ( l ) }$ is the attention ranking of token $t _ { i }$ in layer $l$ and $\sigma ( x )$ denotes the sigmoid function, which amplifies the differences between token rankings by mapping them to a smooth range, ensuring that higher-ranked tokens receive significantly more emphasis Layer Confidence Score Not all layers contribute to the global importance equally. The layer confidence measures the reliability of attention patterns in each layer:

$$
\omega _ { \mathrm { c o n f } } ^ { ( l ) } = \frac { \mu _ { \mathrm { a t t n } } ^ { ( l ) } } { \sigma _ { \mathrm { a t t n } } ^ { ( l ) } + \epsilon }
$$

where µattn and $\sigma _ { \mathrm { a t t n } } ^ { ( l ) }$ attention weights in layer $l$ , respectively. The mean of attention weights reflects the overall strength of focus in a layer, while the standard deviation indicates the consistency of the attention pattern. By combining both, the confidence score rewards layers with strong and stable attention, while penalizing those with high variability or weak responses. Dynamic Updating Mechanism The final importance score $S _ { i }$ for each token $t _ { i }$ is maintained through an exponential moving average across layers:

$$
S _ { i } ^ { ( l ) } = ( 1 - \beta ) \cdot S _ { i } ^ { ( l - 1 ) } + \beta \cdot s _ { i } ^ { ( l ) }
$$

where $\beta$ is the learning rate controlling the update speed, set to 0.2, and $S _ { i } ^ { ( 0 ) } = 0$ is set for initialization.

# 6 Lightweight Action-aware Controller 1 Observation and Insight

In our experiments, when we prune more tokens, we saw a drop in success rate. We observed every frame and found that failure cases occurred during interactions involving object contact, such as manipulating or placing objects in Figure 8(a), and the task merely fails when those actions are successfully executed. Therefore, we emphasize that task success is highly dependent on fine-grained actions, which require high precision and are sensitive to execution errors. In contrast, coarse-grained actions (e.g., moving to a general location) are more tolerant to inaccuracies.

![](images/8.jpg)  

Figure 8: (a) The task process consists of four stages, which can be categorized into coarse-grained and fine-grained actions depending on their respective velocity. (b) Typical failure cases in two fine-grained stages.

Specifically, when the robot approaches an object, the subsequent interaction often demands fine-grained control to ensure stable contact and successful manipulation. Thus, the granularity of the action—whether fine or coarse—plays a critical role in determining the required level of accuracy. Inspired by this observation, we propose an action-aware pruning strategy. By identifying whether a step requires fine or coarse control, our method can selectively preserve more candidates in fine-grained phases while allowing aggressive pruning in coarse-grained ones, improving both efficiency and success rate.

# 6.2 Method

In robotic task execution, we divide the overall process into four phases: targeting, approaching, transferring, and placing. Among these, the object approaching and placing phases involve high-precision operations and are thus classified as fine manipulation stages. During these stages, the translational and rotational velocities of the gripper are generally maintained at low levels, and the velocity along the Input: Full token set $V$ Parameter: Pruning ratio $\alpha$ , Velocity thresholds $v _ { t } ^ { t h } , v _ { r } ^ { t h }$   

Output: Retain token set $V _ { r e t a i n }$ 1: Action-aware Controller 2:Compute translational and rotational velocity $v _ { t }$ and $v _ { r }$ 3: if $v _ { t } < v _ { t } ^ { t h }$ and $v _ { r } < v _ { r } ^ { t h }$ and $\Delta z \le 0$ then 4: Set fine-grained action mode and $K _ { b a s e } = \alpha \times 4 0$ 5:else 6: Set coarse-grained action_mode and $K _ { b a s e } = \alpha \times 2 4$ 7:end if 8: Static Token Pruning at Action Level 9: $V _ { g l o b a l }  \mathrm { t o p } { - } K _ { G }$ tokens from previous action's attention scores   
10: $S i m ( P _ { t } , P _ { t - 1 } )$ bevloci $I _ { t }$ and $I _ { t - T }$ hee $\begin{array} { r } { T = \bar { \lfloor } - \frac { 1 6 } { 3 } \cdot v _ { t } + \frac { 2 2 } { 3 } \rfloor + 4 } \end{array}$   
11: $V _ { d y n a m i c } \gets \mathrm { L o w } – K _ { D }$ patches with similarity $< \tau$   
1 $V _ { l o c a l }  V _ { t o p - K _ { b a s e } } ^ { ( 1 ) } \cup V _ { t o p - K _ { b a s e } } ^ { ( 2 ) }$ wo layers $V _ { r e t a i n }  V _ { g l o b a l } \cup V _ { d y n a m i c } \cup V _ { l o c a l }$   
15: Dynamic Token Pruning at Layer Level 16: if current layer $\mathrm { L }$ in set $L _ { u p d a t e }$ then   
17: $\begin{array} { r } { \omega _ { r a n k , i } ^ { ( l ) }  \frac { \sigma ( - k \cdot \operatorname { r a n k } _ { i } ^ { ( l ) } ) } { \sum _ { j } \sigma ( - k \cdot \operatorname { r a n k } _ { j } ^ { ( l ) } ) } } \end{array}$ {Rank-based weight}   
18: $\omega _ { c o n f } ^ { ( l ) }  \mu _ { a t t n } ^ { ( l ) } / ( \sigma _ { a t t n } ^ { ( l ) } + \epsilon )$ {Layer confidence}   
19: The importance score $s _ { i } ^ { ( l ) } = \omega _ { r a n k , i } ^ { ( l ) } \times \omega _ { c o n f } ^ { ( l ) }$   
20: Update token score: S(l) $S _ { i } ^ { ( l ) } = ( 1 - \beta ) \cdot S _ { i } ^ { ( l - 1 ) } + \beta \cdot s _ { i } ^ { ( l ) }$   
21: end if   
22: if current layer $\mathrm { L }$ in set $L _ { p r u n e }$ then   
23: $V _ { r e t a i n } $ tokens with top $\alpha \times Q \% S _ { i } ^ { ( l ) }$   
24:end if   
25: return $V _ { r e t a i n }$ $z$ -axis is typically negative or smaller than a small positive value, ensuring stability and acuracy. Since each action corresponds to a fixed unit time duration, the robot's policy effectively outputs a normalized velocity command through these relative displacements. We use the displacement magnitude per step as a proxy for instantaneous velocity. The translational velocity $v _ { t }$ is defined as the Euclidean norm of the gripper's 3D relative displacement vector output by the policy at each step:

$$
v _ { t } = \| \Delta x , \Delta y , \Delta z \| _ { 2 } = \sqrt { ( \Delta x ) ^ { 2 } + ( \Delta y ) ^ { 2 } + ( \Delta z ) ^ { 2 } } .
$$

The rotational velocity $v _ { r }$ is computed as the magnitude of the angular displacement:

$$
v _ { r } = \| \Delta \alpha , \Delta \beta , \Delta \gamma \| _ { 2 } = \sqrt { ( \Delta \alpha ) ^ { 2 } + ( \Delta \beta ) ^ { 2 } + ( \Delta \gamma ) ^ { 2 } } ,
$$

expressed in radians. The $z$ axis velocity is represented by the vertical displacement component $\Delta z$ . To formalize this behavior, we define three thresholds: a translational displacement threshold $v _ { t } ^ { \mathrm { t h } }$ , a rotational displacement threshold $v _ { r } ^ { \mathrm { t h } }$ , and a $z$ -axis displacement threshold $v _ { z } ^ { \mathrm { t h } }$ mode when $\bar { v } _ { t } < v _ { t } ^ { \mathrm { t h } } , v _ { r } < v _ { r } ^ { \mathrm { t h } }$ and $\Delta z \le 0$ .

Table 1: Performance Evaluation   

<table><tr><td rowspan="2">Method</td><td colspan="4">Success Rate (%) / Latency (ms) (Speedup)</td><td rowspan="2">Average Speedup</td><td rowspan="2">FLOPs</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td></tr><tr><td>OpenVLA-OFT</td><td>97.6 / 109 (1.00×)</td><td>96.5 / 109 (1.00×)</td><td>97.9 / 109 (1.00×)</td><td>94.5 / 109 (1.00×)</td><td>1.00×</td><td>100%</td></tr><tr><td>SparseVLM</td><td>96.8 / 85.3 (1.28 ×)</td><td>94.2 / 85.3 (1.28×)</td><td>97.6 / 85.3 (1.28×)</td><td>93.6 / 85.3 (1.28×)</td><td>1.28×</td><td>77%</td></tr><tr><td>VLA-Cache</td><td>99.0 / 101 (1.08×)</td><td>97.7 / 102 (1.07×)</td><td>97.4 / 102 (1.07×)</td><td>93.6 / 102 (1.07×)</td><td>1.07×</td><td>83%</td></tr><tr><td>EfficientVLA</td><td>96.5 / 68.8 (1.58×)</td><td>91.1 / 71.4 (1.53×)</td><td>96.0 / 73.7 (1.48×)</td><td>72.1 / 68.6 (1.59×)</td><td>1.55×</td><td>35%</td></tr><tr><td>Ours</td><td>98.2 / 72.4 (1.51 ×)</td><td>96.3 / 76.2 (1.43× )</td><td>97.7 / 73.6 (1.48×)</td><td>94.0 / 78.1 (1.40×)</td><td>1.46×</td><td>43%</td></tr></table>

Table 2: Ablation Study on LIBERO-Spatial   

<table><tr><td></td><td>SR (%)</td><td>Latency (ms)</td><td>Speedup</td></tr><tr><td>None</td><td>97.6</td><td>109</td><td>1.00×</td></tr><tr><td>Tech.1</td><td>97.6</td><td>76.6</td><td>1.42×</td></tr><tr><td>Tech.1&amp;2</td><td>96.8</td><td>70.8</td><td>1.54×</td></tr><tr><td>Tech.1&amp;2&amp;3</td><td>98.2</td><td>72.4</td><td>1.51×</td></tr></table>

Conversely, the system exits the mode when $v _ { t } \mathrm { ~ \textbf ~ { ~ } ~ } >$ $v _ { t } ^ { \mathrm { t h } }$ or $v _ { r } \ > \ v _ { r } ^ { \mathrm { t h } }$ , which typically occurs during object lifting. This strategy ensures early activation of precision control before contact and timely deactivation afterward, effectively balancing operational accuracy with task efficiency in dynamic manipulation scenarios. This strategy ensures the robot initiates fine manipulation mode early during the approach phase, prior to physical contact with the target object. This early activation provides sufficient time for precise positional and orientation adjustments to minimize positioning errors. Subsequent z-axis velocity increases upon successful object contact and lifting trigger an immediate exit from fine mode. The specific control flow is shown in Algorithm 1, which demonstrates the process within one layer of the LLM.

# 7 Experiment

# 7.1 Experimental Settings

Benchmarks and Platforms. We conduct evaluations on the LIBERO simulation benchmark (Liu et al. 2023a), which utilizes a simulated Franka Emika Panda robotic arm. We employ four task suites, LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long, to evaluate the model's capabilities in spatial reasoning, object understanding, goaldirected planning and execution, and long-horizon task completion, respectively. Each task suite consists of ten distinct tasks. All main experiments are conducted on a Linux workstation with an NVIDIA A800-80GB GPU. Implementation Details. We select the OpenVLAOFT (Kim, Finn, and Liang 2025) as the code base for the implementation of our techniques. Due to the varying difficulty and focus of each task suite, we adopt different overall pruning ratios $\alpha$ to balance accuracy and inference speed. Specifically, $\alpha$ is set to 1.0 for LIBERO-Spatial , 0.8 for LIBERO-Goal and 0.6 for LIBERO-Object and LIBEROLong. We also explore the influence of different ratios in design space exploration(DSE) in Table 3. We repeat every task of every task suite for 40 to 50 times to mitigate the impact of random errors. The latency here is defined as the time duration from when the model receives the input to when it generates the action. We report the average inference time over the ten tasks within each task suite. Baselines We select OpenVLA-OFT as our target model. It utilizes DINOv2 (Oquab et al. 2023) and SigLIP (Zhai et al. 2023) as visual encoders to extract visual features, Llama2-7B as the backbone LLM, and a four-layer MLP as an action head to generate continuous actions. The model receives two-view images: the third-person view and the wrist view. We also consider three optimization methods: SparseVLM (Zhang et al. 2024), a framework that adaptively sparsifies less important visual tokens and recycles their information to minimize performance loss, EfficientVLA (Yang et al. 2025), a visual pruning approach for VLA models, and VLA-Cache (Xu et al. 2025b) , which leverages image similarity to cache features across time steps.

# 7.2 Evaluation on Speedup and Success Rate

Table 1 shows the end-to-end evaluation on success rate (SR), latency and speedup on four LIBERO datasets. SpecPrune-VLA achieves an average speedup of $1 . 4 6 \times$ with negligible loss $( < ~ 0 . 7 \% )$ in success rate. The speedup ratio fluctuates slightly because the task difficulty and environment of different datasets vary. VLA-Cache achieves the high success rate on four datasets, but with limited speedup. This is consistent with our analysis that VLA-Cache reduces limited computation. Besides, it also introduces KV cache access latency. EfficientVLA( $_ { \mathrm { L } = 2 8 }$ $_ { \mathrm { T } = 1 1 2 }$ ) achieves higher speedup by skipping four fixed layers and aggressively pruning tokens; however, this leads to a significant drop in success rate in some scenarios, as important tokens and critical action-relevant information in the hidden states are compromised. It is worth noting that the approach is originally for VLA models with a diffusion action expert, which may be more tolerant to the changing output action hidden states.

# 7.3 Ablation Study

To evaluate the effectiveness of our proposed method, we conducted ablation study on the LIBERO-Spatial dataset. Our full model achieves a success rate (SR) of $9 8 . 2 \%$ , which is better than the baseline $( 9 7 . 6 \% )$ . and it significantly reduces latency from 109 milliseconds to 72.4 milliseconds, resulting in a speedup of $1 . 5 1 \times$ compared to OpenVLA

Table 3: Design Space Exploration on Pruning Ratio   

<table><tr><td>Prune Ratio (α)</td><td>Pruned Tokens</td><td>SR (%)</td><td>Latency(ms) / Speedup</td></tr><tr><td>1.0</td><td>346</td><td>83.7</td><td>71.9 / 1.52×</td></tr><tr><td>0.8</td><td>329</td><td>95.2</td><td>74.8 / 1.46×</td></tr><tr><td>0.6</td><td>318</td><td>96.3</td><td>76.2 / 1.43 ×</td></tr></table>

![](images/9.jpg)  

Figure 9: Evaluation on NVIDIA GeForce RTX 3090

OFT. This demonstrates the efficiency gains of our approach while maintaining competitive accuracy. The ablation study further highlights the importance of each component: Static(Tech 1) and Dynamic(Tech 2) pruning slightly affect the SR $( 9 6 . 8 \% )$ but reduces latency to 70.8 milliseconds, indicating that pruning contributes to the overall latency reduction. The introduction of the action adapter increases the success rate and cause negligible latency(1.6ms). This suggests that the action adapter plays a crucial role in maintaining high accuracy.

# 7.4 Design Space Exploration

We conduct a design space exploration on the LIBEROObject. To reduce the model latency while maintaining accuracy, we focus on the number of pruned tokens. We set a prune ratio $\alpha$ to adjust both static token pruning and dynamic token pruning. The bigger the prune ratio, the more tokens are pruned, leading to a drop in success rate and a rise in speedup. To balance accuracy and speed, we set the prune ratio to 0.6 for LIBERO-Object.

# 7.5 Evaluation on Various Computing Platforms

To validate the applicability of our method on different devices, we conduct experiments on NVIDIA GeForce RTX 3090. As illustrated in Figure 9, our method achieves an average speedup of $2 . 0 9 \times$ in LLM inference time and $1 . 5 7 \times$ in end-to-end latency across all task categories using the same parameters as Table 1. The results consistently demonstrate improved inference efficiency, underscoring the scalability and effectiveness of our approach under diverse computational conditions.

# 7.6 Experiment Visualization

To understand what the model actually observed when completing a task, we visualize the retained visual tokens. The colored patches are the retained tokens. We show the visualization results of three tasks in Figure 10. The results show that retained tokens are the important tokens discussed in previous sections. Moreover, the method works when processing images from different viewpoints, demonstrating its ability to capture viewpoint-invariant representations. Task: pick up the alphabet soup and place it in the basket

![](images/10.jpg)  

Figure 10: Retained tokens during tasks

# 8 Conclusion

In this paper, we propose a novel insight that combines local information in the current action generation with global information in previous actions for token selection, and present SpecPrune-VLA, a training-free pruning method including two-level token pruning with heuristic control. Experiments show that SpecPrune-VLA achieve an average $1 . 4 6 \times$ speedup on NVIDIA A800-80GB and $1 . 5 7 \times$ speedup on NVIDIA GeForce RTX 3090, with negligible degradation in task success rate on the LIBERO simulation benchmark. The evaluation demonstrates that our method generalizes well across different hardware platforms.

# 9 Limitation

One limitation of this work is that all experiments are conducted in simulated environments. While simulation enables controlled and scalable evaluation, real-world deployment may introduce additional challenges such as sensor noise, environmental dynamics, and hardware latency. We acknowledge this gap and plan to deploy our method on physical platforms in the future when opportunities arise, to validate its effectiveness in real-world scenarios.

References   
Black, K.; Brown, N.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; Groom, L.; Hausman, K.; Ichter, B.; et al. 2024. $\pi _ { 0 }$ : A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164. Brohan, A.; Brown, N.; Carbajal, J.; Chebotar, Y.; Dabis, J.; Finn, C.; Gopalakrishnan, K.; Hausman, K.; Herzog, A.; Hsu, J.; et al. 2022. Rt-1: Robotics transformer for realworld control at scale. arXiv preprint arXiv:2212.06817. Dong, Y.; Miao, Y.; Li, W.; Zheng, X.; Wang, C.; and Lyu, F. 2025. Accelerating llm inference throughput via asynchronous kv cache prefetching. arXiv preprint arXiv:2504.06319.   
Elhoushi, M.; Shrivastava, A.; Liskovich, D.; Hosmer, B.; Wasti, B.; Lai, L.; Mahmoud, A.; Acun, B.; Agarwal, S.; Roman, A.; et al. 2024. LayerSkip: Enabling early exit inference and self-speculative decoding. ariv preprint arXiv:2404.16710.   
He, Z.; Yao, Y.; Zuo, P.; Gao, B.; Li, Q.; Zheng, Z.; and Wu, F. 2025. Adaskip: Adaptive sublayer skipping for accelerating long-context llm inference. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 2405024058.   
Kim, M. J.; Finn, C.; and Liang, P. 2025.Fine-u vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645.   
Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E.; Lam, G.; Sanketi, Pe al. 0.Openvla:An pen-source vision-angge action model. arXiv preprint arXiv:2406.09246.   
Leviathan, Y.; Kalman, M.; and Matias, Y. 2023. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, 1927419286. PMLR.   
Li, Q.; Liang, Y.; Wang, Z.; Luo, L.; Chen, X.; Liao, M.; Wei, F.; Deng, Y.; Xu, S.; Zhang, Y.; et al. 2024.Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650.   
Li, Y.; Meng, Y.; Sun, Z.; Ji, K.; Tang, C.; Fan, J.; Ma, X.; Xia, S.; Wang, Z.; and Zhu, W. 2025. SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration. arXiv preprint arXiv:2506.12723. L B. ZuY.; Gao, C.; FengY.; Liu Q.;ZhuY.and Stone, P. 2023a. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36: 4477644791.   
Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023b. Visual instruction tuning. Advances in neural information processing systems, 36: 3489234916.   
Liu, H.; Li, X.; Li, P.; Liu, M.; Wang, D.; Liu, J.; Kang, B.; Ma, X.; Kong, T.; and Zhang, H. 2025. Towards generalist robot policies: What matters in building vision-languageaction models.   
L J.; Liu M. Wa Z.;Lee, L ZuK. An, P S.; Zhang, R.; Guo, Y.; and Zhang, S. 2024. Robomamba: Multial   oel  eft oo and manipulation. arXiv e-prints, arXiv2406.   
Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; ElNouby, A.; et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193. O'Neill, A.; Rehman, A.; Maddukuri, A.; Gupta, A.; Paakar, A.; Lee A.; Pooley, A.; Gupta A.;Madekar A.; Ji A.;t al. 0. Ope x-odimet: Roboic lar datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), 68926903. IEEE.   
Park, S.; Kim, H.; Jeon, W.; Yang, J.; Jeon, B.; Oh, Y.; and Choi, J. 2024. Quantization-aware imitation-learning for resource-efficient robotic control. arXiv preprint arXiv:2412.01034.   
Peebles, W.; and Xie, S. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 41954205.   
Sapkota, R.; Cao, Y.; Roumeliotis, K. I.; and Karkee, M.2025. Vision-language-action models: Concepts, progress, applications and challenges. arXiv preprint arXiv:2505.04769.   
Team, O. M.; Ghosh, D.; Walke, H.; Pertsch, K.; Black, K.; Mees, O.; Dasari, S.; Hejna, J.; Kreiman, T.; Xu, C.; et al. 2024. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213.   
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2023. Attention Is All You Need. arXiv:1706.03762.   
Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Fan, Y.; Dang, K.; Du, M.; Ren, X.; Men, R.; Liu, D.; Zhou, C.; Zhou, J. and Lin, J. 2024. Qwen2-VL: Enhancing VisionLanguage Model's Perception of the World at Any Resolution. arXiv:2409.12191.   
We, J.; Zhu, Y.; ZhuM. Tng Z.;Li J.;ZouZ.Lu, X.; Shen, C.; Peng, Y.; and Feng, F. 2025. DiffusionVLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression. In Forty-second International Conference on Machine Learning.   
Xu, J.; Pan, J.; Zhou, Y.; Chen, S.; Li, J.; Lian, Y.; Wu, J.; and Dai, G. 2025a. Specee: Accelerating large language model inference with speculative early exiting. In Proceedings of the 52nd Annual International Symposium on Computer Architecture, 467481.   
Xu, S.; Wang, Y.; Xia, C.; Zhu, D.; Huang, T.; and Xu, C. 2025b. Vla-cache: Towards efficient vision-language-action model via adaptive token caching in robotic manipulation. arXiv preprint arXiv:2502.02175.   
Ya Y.; Wang Y.; Wen, Z. Zhi, L; Zou, C.; Zh, Z. Wen, C.; and Zhang, L. 0.EfficLA: Trai Free Acceleration and Compression o ision-LanguageAction Models. arXiv preprint arXiv:2506.10100.   
Yue, Y.; Wang, Y.; Kang, B.; Han, Y.; Wang, S.; Song, S.; Feng, J.; and Huang, G. 2024. Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution. Advances in Neural Information Processing Systems, 37: 5661956643.   
Zhai, X.; Mustafa, B.; Kolesnikov, A.; and Beyer, L. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, 1197511986.   
Zhang, Y.; Fan, C.-K.; Ma, J.; Zheng, W.; Huang, T.; Cheng, K.; Gudovskiy, D.; Okuno, T.; Nakata, Y.; Keutzer, K.; et al. 2024. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417.   
Zhao, T. Z.; Kumar, V.; Levine, S.; and Finn, C. 2023. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705.   
Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; et al. 2023. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, 2165 2183. PMLR.

# 10 Appendix

# 10.1 Complexity Analysis

Consider a Transformer model with $N$ layers. The computational cost (in FLOPs) per layer when processing $L$ tokens is approximately:

$$
\mathrm { F L O P s } _ { \mathrm { l a y e r } } ( L ) = 4 L D ^ { 2 } + 2 L ^ { 2 } D + 2 L D M
$$

where: - $L$ : sequence length (number of tokens) - $D$ : hidden dimension - $M$ : number of attention heads Static Token Pruning The static token pruning strategy reduces an average of 315 tokens from the original about 600 tokens, let $L _ { r } = \alpha \cdot L$ denote the number of retained tokens, with $\alpha = 0 . 4 8$ .

<table><tr><td>Benchmark</td><td></td><td></td><td></td><td></td><td>Spatial Goal Object Long Average</td></tr><tr><td>Pruned tokens</td><td>333</td><td>326</td><td>318</td><td>304</td><td>315</td></tr></table>

Table 4: Number of pruned tokens across different LIBERO datasets under static pruning. The pruned number slightly changes as the environment changes.

Thus, for most layers, the FLOPs become $\mathrm { F L O P s } ( L _ { r } )$ . Dynamic Token Pruning We apply an extra $10 \%$ pruning at the target layer:

$$
\mathcal { S } = \{ 5 , 1 0 , 1 5 , 2 0 \}
$$

Therefore, the number of retained tokens after these layers becomes $L _ { r ^ { \prime } } = 0 . 9 \cdot L _ { r }$ On average, this reduces the token count by $1 9 \%$ across all layers. Overall FLOPs reduction Hence, the overall FLOPs reduction is:

$$
\Delta \mathrm { F L O P s } = ( 1 - \frac { 3 0 } { 3 2 } \times 0 . 4 8 \times 0 . 8 1 ) \times \mathrm { F L O P s } \approx 0 . 6 3 \mathrm { F L O P s }
$$

# 10.2 Experiment Details

# Implementation of Comparative Methods

We provide details on the implementation of the baseline and comparative methods. All models were implemented on one A800 GPU. OpenVLA-OFT The latency and the success rate on four datasets are similar to the data in the original paper. VLA-Cache While the original method was developed on the OpenVLA model, its authors adapted and extended it to the OpenVLA-OFT. All results reported in our experiments are obtained using the authors' official implementation to ensure reproducibility. EfficientVLA The method focuses on VLA models with diffusion action expert and it also optimizes the action expert. Besides, it has not been open-sourced. As a result, we re-implement it according to the details of visual token pruning and layer pruning provided in the original paper. Following the reported setup, we retain 28 transformer layers and 56 visual tokens throughout inference.

# Experimental Setup

In the static token pruning stage, the number of tokens preserved from both the global top- $K$ tokens (from the previous inference step) and the current top- $K$ tokens is controlled by a base threshold. We set the base threshold to $K _ { c \mathrm { ~ } } = \mathrm { \ A T T N _ { - } T O P K ~ = ~ 2 4 ~ }$ for coarse-grained mode and $K _ { p } = \mathrm { A T T N \mathrm { { . T O P K } \mathrm { { . P R E C I S E } = 4 0 } } }$ for fine-grained mode. The actual number of retained tokens is scaled by the pruning rate $\alpha$ , such that:

$$
K _ { \mathrm { r e t a i n } } = \alpha \times { \left\{ \begin{array} { l l } { K _ { c } , } & { { \mathrm { c o a r s e - g r a i n e d ~ m o d e } } , } \\ { K _ { p } , } & { { \mathrm { f i n e - g r a i n e d ~ m o d e } } . } \end{array} \right. }
$$

We apply different $\alpha$ values depending on the dataset to balance efficiency and performance: - LIBERO-Long: $\alpha = 1 . 0$ - LIBERO-Goal: $\alpha = 0 . 8$ - LIBERO-Object: $\alpha = 0 . 6$ - LIBERO-Spatial: $\alpha = 0 . 6$

# Experiment Visualization

To understand what the model actually observed when completing a task, we visualize the retained visual tokens. The colored patches are the retained tokens. We show the visualization results of four tasks on four LIBERO datasets in Figure 11. It shows the retained tokens are the task-relevant and dynamic tokens.

# 10.3 Further Analysis

Different camera viewpoints present distinct characteristics. In this section, we provide a systematic exposition of the first key insight. In the fixed third-person view, the robot arm and objects it touches are the dynamic component, while the taskrelevant patches (e.g., objects on the table) are relatively static, as Figure 11 shows. Therefore, the key is to extract the intersection of both — the regions that involve the dynamic and the task-relevant pixels. In the wrist-mounted camera view, although all objects are in motion, the pixel patches along object boundaries exhibit more significant changes. We process and interpret the temporal signal of a pixel patch using the Fourier transform:

$$
P ( f ) = \int _ { - \infty } ^ { \infty } p ( t ) e ^ { - j 2 \pi f t } d t
$$

Here $p ( t )$ represents pixel intensity at time $t$ , $P ( f )$ denotes the frequency-domain representation of the signal, $f$ stands for frequency. Since objects have different colors from the background, the boundary patches will change color at a certain time. Therefore, in these regions, the resulting signal $p ( t )$ contains more abrupt changes, which correspond to higher energy in the high-frequency components of $P ( f )$ . This indicates that boundary patches carry richer temporal dynamics. As a result, task-relevant patches and dynamic patches can be complementary to some extent, and their intersection can also be used to identify the most critical regions.

![](images/11.jpg)  

Task: pick up the black bowl between the plate and the ramekin and place it on the plate LIBERO-Spatial

![](images/12.jpg)  

Task: open the middle drawer of the cabinet LIBERO-Goal

![](images/13.jpg)  

Task: put the yellow and white mug in the microwave and close it

![](images/14.jpg)  
LIBERO-Long   

Figure 11: Retained tokens during tasks across four datasets and different views.