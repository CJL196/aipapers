# Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning

Qi Wang1,2,3\* Zhipeng Zhang4,5\* Baao Xie2,3\* Xin Jin2,3 Yunbo Wang1 Shiyu Wang5,6 Liaomo Zheng5,6 Xiaokang Yang1 Wenjun Zeng2,3   
1 MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University 2 Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo, China   
3 Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Ningbo, China 4 University of Chinese Academy of Sciences 5 Shenyang Institute of Computing Technology, Chinese Academy of Sciences 6 Shenyang CASNC Technology Co., Ltd https://qiwang067.github.io/diswm

# Abstract

Training visual reinforcement learning (RL) in practical scenarios presents a significant challenge, i.e., RL agents suffer from low sample efficiency in environments with variations. While various approaches have attempted to alleviate this issue by disentangled representation learning, these methods usually start learning from scratch without prior knowledge of the world. This paper, in contrast, tries to learn and understand underlying semantic variations from distracting videos via offline-to-online latent distillation and flexible disentanglement constraints. To enable effective cross-domain semantic knowledge transfer, we introduce an interpretable model-based RL framework, dubbed Disentangled World Models (DisWM). Specifically, we pretrain the action-free video prediction model ofline with disentanglement regularization to extract semantic knowledge from distracting videos. The disentanglement capability of the pretrained model is then transferred to the world model through latent distillation. For finetuning in the online environment, we exploit the knowledge from the pretrained model and introduce a disentanglement constraint to the world model. During the adaptation phase, the incorporation of actions and rewards from online environment interactions enriches the diversity of the data, which in turn strengthens the disentangled representation learning. Experimental results validate the superiority of our approach on various benchmarks.

![](images/1.jpg)  
Figure 1. Overview of our proposed framework. The key idea is to leverage distracting videos for semantic knowledge transfer, enabling the downstream agent to improve sample efficiency on unseen tasks.

# 1. Introduction

Visual reinforcement learning (VRL) presents a promising approach for training agents within complex environments [11, 18, 20, 28, 45]. However, VRL frequently suffers from performance degradation in practical scenarios due to the complexity, volatility, and visual distractions in environments. Even minor environmental variations can result in significant pixel-level shifts, making the trained VRL policies ineffective or suboptimal [6, 7]. For instance, a slight change in lighting conditions can affect an object's appearance (e.g., color, shadow, or other visual attributes). Therefore, it is crucial to enhance models with interpretability, enabling them to perceive, learn, and understand the semantic environmental variations.

Disentangled representation learning (DRL) presents a promising approach to addressing the interpretability challenges inherent in the "black-box" nature of deep learning algorithms. Fundamentally, DRL approaches mimic the cognitive processes of biological intelligence, where understanding the world is facilitated by decomposing observations into distinct and independent factors [2, 14, 38, 40, 41]. In this form, when a factor of variation is changed (e.g., color), only a small portion of features in the disentangled representation will be affected, enabling the agent to recover performance quickly. Several studies have explored the integration of DRL algorithms in the domain of VRL. For example, Higgins et al. [15] trained a $\beta$ VAE offline to obtain disentangled representations for reinforcement learning. TED [7] adopts a self-supervised auxiliary task to learn temporally disentangled representations for reinforcement learning. Additionally, Dunion et al. [6] introduced conditional mutual information to achieve a disentangled representation with correlated data. However, existing methods typically learn the representations from scratch, lacking any prior knowledge of the world. These approaches often require extensive interactions with the environment to acquire desired behaviors.

Table 1. MuJoCo (downstream domain) vs. DMC (accessible distracting videos).   

<table><tr><td></td><td>Video: DMC</td><td>Target: MuJoCo</td><td>Similarity / Difference</td></tr><tr><td>Task</td><td>Reacher Easy</td><td>Pusher</td><td>Relevant robotic control tasks</td></tr><tr><td>Dynamics</td><td>Two-link planar</td><td>Multi-jointed robot arm</td><td>Different</td></tr><tr><td>Action space</td><td>Box(-1, 1, (2,), float32)</td><td>Box(-2, 2, (7,), float32)</td><td>Different</td></tr><tr><td>Reward range</td><td>[0, 1]</td><td>[-4.49, 0]</td><td>Different</td></tr></table>

Towards this challenge, we introduce a model-based interpretable VRL framework, dubbed Disentangled World Models (DisWM), which leverages prior knowledge extracted from distracting videos to facilitate the learning of unseen downstream tasks through latent distillation. It is crucial to note that distracting videos refer to videos with visual distractions, which are beneficial for learning disentangled representations. Specifically, as depicted in Figure 1, our framework consists of two phases: first, we pretrain a DRL encoder to learn disentangled latent representations from distracting videos. By doing so, the pretrained DRL encoder is "knowledgeable" in terms of representation disentanglement. Subsequently, we finetune an orthogonally designed world model with dual constraints of disentanglement and distillation, leveraging semantic knowledge transferred from the pretrained model via offline-to-online latent distillation. Another benefit of disentangled world model adaptation is that incorporating actions and rewards from online interactions with the environment enriches the diversity of the visual observations, which in turn strengthens the process of disentangled representation learning. It is worth mentioning that, as a cross-domain framework, DisWM does not require the pretraining videos to originate from the same domain as the downstream tasks. Experimental results demonstrate the effectiveness of our proposed approach in improving the sample efficiency of VRL agents across our modified DeepMind Control and MuJoCo Pusher. The contributions of this work can be summarized as follows: • We frame the problem of learning interpretable VRL agents as a domain transfer learning problem. The key idea is to extract semantic knowledge from distracting videos and transfer this disentanglement capability to downstream control tasks. • We present DisWM, an approach that follows the pretraining-finetuning paradigm using distracting videos, incorporating specific techniques of ofline-to-online latent distillation and flexible disentanglement constraints.

# 2. Problem Setup

We formulate visual reinforcement learning as a partially observable Markov decision process (POMDP) that uses DMC and MuJoCo Pusher as the test bench. Specifically, we concentrate on scenarios where videos without actions and rewards are accessible, enabling world knowledge transfer. The goal is to maximize the cumulative reward of the target POMDP $\langle \mathcal { O } , \mathcal { A } , \mathcal { T } , \mathcal { R } , \gamma \rangle$ by transferring the shared world knowledge from the videos. These notations correspond to the visual observation space, the action space, the transition probabilities, the reward function, and the discount factor, respectively. For instance, in one of the cross-domain experiments, we use MuJoCo as the downstream domain and the frames collected from DMC as the distracting video. Table 1 highlights the differences between the two domains in terms of visual appearances, physical dynamics, action spaces, and reward functions.

# 3. Method

# 3.1. Overview of DisWM

In this section, we present the details of DisWM, which involves three main stages (see Figure 2): a) Disentangled representation pretraining: Pretrain a DRL-based video prediction model from distracting videos to extract disentangled features.   
b) Offine-to-online latent distillation: Transfer the semantic knowledge from the pretrained model to the world model via cross-domain latent distillation.   
c) Disentangled world model adaptation: Finetune the downstream agent with disentanglement constraints by incorporating the action and reward information.

![](images/2.jpg)  
trained on distracting videos offine for the wel-disentangled latent varable $\mathbf { z } _ { \mathrm { d i s e n } }$ , which extracts semantic knowledge from the visual observations. The disentangled capability of $\mathbf { z } _ { \mathrm { d i s e n } }$ is then transferred to the world model through latent distillation. (b) The action-conditioned

# 3.2. Disentangled Representation Pretraining

To extract well-disentangled representations that can be transferred to the downstream world model, we first train a video prediction model on distracting videos without incorporating action information (Lines 4-8 of Alg. 1). This model comprises three key components: (i) the posterior learner that encodes the observation $o _ { t }$ to latent state $z _ { t }$ via $\beta$ -VAE encoder1, which serves as a typical DRL framework to extract latent features $\mathbf { z } _ { t }$ from observations, (ii) the prior module that predicts future latent states based on historical states, without directly relying on the current observation $o _ { t }$ , and (iii) the $\beta$ -VAE-based decoder that reconstructs $\hat { o } _ { t }$ from the latent state $z _ { t }$ . Concretely, the model can be formulated as follows:

$\beta$ VAE encoder: $\begin{array} { r l } & { \mathbf { z } _ { t } = e _ { \phi ^ { \prime } } ( o _ { t } ) } \\ & { z _ { t } \sim q _ { \phi ^ { \prime } } ( z _ { t } \mid z _ { t - 1 } , \mathbf { z } _ { t } ) } \\ & { \hat { z } _ { t } \sim p _ { \phi ^ { \prime } } ( \hat { z } _ { t } \mid z _ { t - 1 } ) } \\ & { \hat { o } _ { t } \sim p _ { \phi ^ { \prime } } ( \hat { o } _ { t } \mid z _ { t } ) } \end{array}$ Posterior state:   
Prior state:   
Reconstruction: Isotropic unit Gaussian: $p ( \mathbf { z } ) = \mathcal { N } ( \mathbf { 0 } , I )$ . where $\phi ^ { \prime }$ denotes the parameters of the model. The $\beta$ VAEbased video prediction model is trained to minimize the fol lowing loss function:

$$
\begin{array} { r l } & { \mathcal { L } ( \phi ^ { \prime } ) = \mathbb { E } _ { q _ { \phi ^ { \prime } } } \Big [ \displaystyle \sum _ { t = 1 } ^ { T } \underbrace { - \ln p _ { \phi ^ { \prime } } ( o _ { t } \mid z _ { t } ) } _ { \mathrm { i m a g e r e c o n s t u c t i o n } } } \\ & { \qquad + \underbrace { \beta _ { 1 } \mathrm { K L } [ q _ { \phi ^ { \prime } } ( \boldsymbol { z } _ { t } \mid \boldsymbol { z } _ { t - 1 } , \boldsymbol { o } _ { t } ) \| p _ { \phi ^ { \prime } } ( \hat { \boldsymbol { z } } _ { t } \mid \boldsymbol { z } _ { t - 1 } ) ] } _ { \mathrm { a c t i o n . f r e K L l o s s } } } \\ & { \qquad + \beta _ { 2 } \mathrm { K L } [ q _ { \phi ^ { \prime } } ( \mathbf { z } _ { t } \mid \boldsymbol { o } _ { t } ) \| p ( \mathbf { z } _ { t } ) \| ] . } \end{array}
$$

The variantional posterior distribution $q _ { \phi ^ { \prime } } ( \mathbf { z } _ { t } \mid \mathbf { \theta } _ { o _ { t } } )$ is encouraged to be close to the standard multivariate Gaussian distribution $\mathcal { N } ( \mathbf { 0 } , I )$ to strengthen the orthogonality and disentanglement of the latent space. The importance of the disentanglement loss term is governed by $\beta _ { 2 }$ .

# 3.3. Offline-to-Online Latent Distillation

After the offline pretraining with the distracting videos, the model is fine-tuned online to adapt to the downstream task by integrating actions and rewards (Lines 13 of Alg. 1). A straightforward approach to transfer disentangled features involves initializing the action-conditioned world model with checkpoints obtained from the pretrained video prediction model. Nevertheless, it may experience a potential mismatch issue caused by the discrepancies between the two domains in visual appearances and physical dynamics. Directly applying the pretraining-finetuning paradigm for downstream tasks tends to overwrite the disentangled information encoded in the pretrained latent features, leading to decreased performance when there are large domain discrepancies between the source and the target domains.

Through comprehensive pretraining on distracting videos that contain diverse visual variations, the video prediction model thus builds an interpretable and orthogonality latent space. In this space, the latent variable $\mathbf { z } _ { \mathrm { d i s e n } }$ achieves a high degree of disentanglement. To exploit the prior semantic knowledge from the pretrained model and improve the sample efficiency of the downstream tasks, we introduce an offline-to-online latent distillation. This approach enables the disentangling capability of $\mathbf { z } _ { \mathrm { d i s e n } }$ from the pretrained model to be effectively transferred to the latent variable $\mathbf { z } _ { \mathrm { t a s k } }$ of the world model. Specifically, this is achieved by minimizing the Kullback-Leibler (KL) divergence between the latent distributions of the two domains. The corresponding distillation loss ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ can be formulated as follows:

$$
{ \mathcal { L } } _ { \mathrm { d i s t i l l } } = \mathrm { K L } \left( \mathbf { z } _ { \mathrm { d i s e n } } \| \mathbf { z } _ { \mathrm { t a s k } } \right) = \sum \mathbf { z } _ { \mathrm { d i s e n } } \cdot \log \left( { \frac { \mathbf { z } _ { \mathrm { d i s e n } } } { \mathbf { z } _ { \mathrm { t a s k } } } } \right)
$$

# 3.4. Disentangled World Model Adaptation

By obtaining well-disentangled representations of $\mathbf { z } _ { \mathrm { d i s e n } }$ and employing the latent distillation for knowledge transfer, we then propose a DRL-based world model $\mathcal { M } _ { \phi }$ , designed to harness these features to enhance interoperability and robustness against environmental variations (Lines 14-15 of Alg. 1). The components of $\mathcal { M } _ { \phi }$ can be detailed as follows:

Recurrent transition: $\begin{array} { l } { h _ { t } = f _ { \phi } ( h _ { t - 1 } , z _ { t - 1 } , a _ { t - 1 } ) } \\ { \mathbf { z } _ { t } \sim e _ { \phi } ( o _ { t } ) } \\ { z _ { t } \sim q _ { \phi } ( z _ { t } \mid h _ { t } , \mathbf { z } _ { t } ) } \\ { \tilde { z } _ { t } \sim p _ { \phi } ( \tilde { z } _ { t } \mid h _ { t } ) } \\ { \tilde { \phi } _ { t } \sim p _ { \phi } ( \hat { o } _ { t } \mid h _ { t } , z _ { t } ) } \\ { \hat { r } _ { t } \sim r _ { \phi } ( \hat { r } _ { t } \mid h _ { t } , z _ { t } ) } \\ { \tilde { \gamma } _ { t } \sim p _ { \phi } ( \hat { r } _ { t } \mid h _ { t } , z _ { t } ) } \end{array}$   
$\beta$ VAE encoder:   
Posterior state:   
Prior state:   
Reconstruction:   
Reward prediction:   
Discount factor: Isotropic unit Gaussian: $p ( \mathbf { z } ) = \mathcal { N } ( \mathbf { 0 } , I )$ , where $\phi$ represents the combined parameters of the world model. We train $\mathcal { M } _ { \phi }$ on the sampled data from the replay buffer $\boldsymbol { B }$ with the following loss function:

$$
\begin{array} { r l } & { \mathcal { L } ( \phi ) = \mathbb { E } _ { q _ { \phi } } \Big [ \displaystyle \sum _ { t = 1 } ^ { T } \underbrace { - \ln p _ { \phi } ( o _ { t } \mid h _ { t } , z _ { t } ) } _ { \mathrm { i m g e r e c o n s t u c i o n } } \underbrace { - \ln r _ { \phi } ( r _ { t } \mid h _ { t } , z _ { t } ) } _ { \mathrm { r e w a r d p r e d i c i o n } } } \\ & { \qquad \underbrace { - \ln p _ { \phi } ( \gamma _ { t } \mid h _ { t } , z _ { t } ) } _ { \mathrm { d i s c o u n p r e d i c i o n } } \underbrace { + \alpha \mathrm { K L } \left[ q _ { \phi } ( z _ { t } \mid h _ { t } , o _ { t } ) \right] \mid p _ { \phi } ( \hat { z } _ { t } \mid h _ { t } ) \Big ] } _ { \mathrm { K L d i v e r g e n c e } } } \\ & { \qquad \underbrace { + \beta \mathrm { K L } \left[ q _ { \phi } ( \mathbf { z } _ { t } \mid o _ { t } ) \mid \mid p ( \mathbf { z } _ { t } ) \right] } _ { \mathrm { . . . . . . ~ . ~ . } } + \underbrace { \eta \mathcal { L } _ { \mathrm { d i s s i n } } } _ { \mathrm { . . . . . . ~ . } } \Big ] . } \end{array}
$$

where $\beta$ is a hyperparameter used to balance reconstruction quality and disentanglement capability. In this adaptation stage, $\eta$ serves as a hyperparameter that gradually decreases from 0.1 to 0.01. Intuitively, $\eta$ controls the progressive adaptation of the world model with the shared world knowledge transfer from the pretrained video prediction model. Through the comprehensive training processes of this framework, we equip the DisWM with the capability to learn and understand the underlying semantic representations. This enhancement enables the model to be less sensitive to environmental variations, such as changes in object colors, positions, and backgrounds. Furthermore, by incorporating actions and rewards during the finetuning phase, the world model can generate data with more diverse representations, thereby improving the disentangled representation learning. For behavior learning, we utilize the actorcritic method that is in line with DreamerV2 [10] (Lines 16-18 of Alg. 1). For more details of behavior learning, please refer to Supplementary Material B.

# 4. Experiments

# 4.1. Experimental Setups

Benchmark We evaluate DisWM on DeepMind Control Suite (DMC) [34], MuJoCo Pusher [35], and DrawerWorld [37]. DMC is a widely adopted benchmark with comprehensive and flexible robotic-control tasks. For DMC benchmark, we use 5 tasks, i.e., Walker Walk, Cheetah Run, Hopper Stand, Finger Spin, Cartpole Swingup. In MuJoCo Pusher, a multi-jointed robotic arm is employed to manipulate a target cylinder (object). The goal is to move the object to a designated target position using the robot's end effector (fingertip). The agent receives the negative reward, which is a combination of three components: the distance between the fingertip and the target, the distance between the object and the target goal position, and a penalty for large actions. DrawerWorld is a modified Metaworld [43] benchmark designed to evaluate texture adaptability in manipulation tasks. It includes five additional textures from realistic photos and grid texture. During training, we initially employ the grid texture and then change to the wood texture midway, while adopting the metal texture exclusively for evaluation. The corresponding results on DrawerWorld are reported in Supplementary Material C.2. Furthermore, the introduction of compared baselines is given in Supplementary Material A. Implementation details. The visual observations of online finetuning stages are resized to $6 4 \times 6 4$ pixels. Inspired by APV [29], we build distracting video datasets with 1M frames using DreamerV2 [10] to interact with the environments with visual color distractors. This video datasets consist of the samples stored in the replay buffer throughout the

# Algorithm 1 The training pipeline of DisWM.

1: Hyperparameters: $H$ : Horizon of latent imagination.

Require: Distracting video dataset $\mathcal { D }$ .   
3: Initialize: Parameters of the model $\{ \phi , \psi , \xi \}$ .   
4 for training step $t = 1 , 2 , \ldots , K _ { 1 }$ do Disentangled representation pretraining   
5: Sample random minibatch $\{ o _ { t } \} _ { t = 1 } ^ { T } \sim \mathcal { D }$ .   
6: Obtain Gaussian prior $\mathbf { z } _ { t }$ from Isotropic unit Gaussian $\mathcal { N } ( \mathbf { 0 } , I )$ .   
7: Pretrain the action-free video prediction model with disentanglement regularization by minimizing Eq. (2).   
8:end for   
9: Train the random agent and collect a replay buffer $\boldsymbol { B }$ .   
10: while not converged do   
11: for training step $t = 1 , 2 , \ldots , K _ { 2 }$ do   
12: Sample $\{ ( o _ { t } , a _ { t } , r _ { t } ) \} _ { t = 1 } ^ { T } \sim \{$ .   
13: Distill the disentangled features to the world model using Eq. (3). Offline-to-online latent distillation   
14: Obtain Gaussian prior $\mathbf { z } _ { t }$ from Isotropic unit Gaussian $\mathcal { N } ( \mathbf { 0 } , I )$ . $\triangleright$ Disentangled world model adaptation   
15: Train the world model $\mathcal { M } _ { \phi }$ with latent distillation and disentanglement constraints using Eq. (5).   
16: Geenerate $\{ ( \hat { z } _ { i } , \hat { a } _ { i } ) \} _ { i = t } ^ { t + H }$ using $\pi _ { \psi }$ and $\mathcal { M } _ { \phi }$ Behavior learning   
17: Train the critic $v _ { \xi }$ ovver $\{ ( \hat { z } _ { i } , \hat { a } _ { i } ) \} _ { i = t } ^ { t + H }$   
18: Train  the actor $\pi _ { \psi }$ over $\{ ( \hat { z } _ { i } , \hat { a } _ { i } ) \} _ { i = t } ^ { t + H }$ .   
19: end for   
20: $o _ { 1 } \gets \in \mathrm { n v }$ .reset() Environment interaction   
21: for time step $t = 1 , 2 , \dots , T$ do   
22: Sample $\hat { a } _ { t } \sim \pi _ { \psi } \big ( \hat { a } _ { t } \mid \hat { z } _ { t } \big )$ .   
23: rt,Ot+1 ←env.step $( \hat { a } _ { t } )$ .   
24: end for   
25: Append data to the replay buffer $\boldsymbol { B }$ .   
26end while training process until the agent achieves maximum score. For tasks in the DMC benchmark, the training steps of the agent are limited to $1 \times 1 0 ^ { 6 }$ environment steps. Each run of DisWM requires roughly 5GB of VRAM and takes around 16 hours to train on a single RTX 3090 GPU. The dimensions of well-disentangled latent $\mathbf { z } _ { \mathrm { d i s e n } }$ and downstream task latent $\mathbf { z } _ { \mathrm { t a s k } }$ are both set to 20 in our approach. In Figure 3, we showcase the example observations of various tasks with color distractors. We train the agent using a fixed set of colors, where the RGB values are varied within a restricted range around the original values. Furthermore, at the midpoint of the training process, we change to a different color scheme for varying distractors.

# 4.2. Main Comparison

We evaluate the sample efficiency and task performance for all the methods with the training curves of episode return. Figure 4 shows the performance of DisWM and all the baselines. Remarkably, it achieves better performance than TED [7], a method on top of RAD [17] and tailored for environments with distractors. For the offline-to-online finetuning models, DV2 Finetune achieves the second-best performance by transferring knowledge from the distracting videos. However, we observe a significant decline in sample efficiency, particularly in scenarios with large data distribution shifts between the source and target domains (e.g., $\mathrm { D M C }  \mathrm { M u J o C o ) }$ . These shifts can occur in various aspects, including visual observation, physical dynamics, reward definition, or the action space of the robots. Another crucial baseline is APV [29], which focuses on transferring knowledge obtained from videos with a stacked latent prediction model. Nevertheless, without environment-specific designs for visual distractors, directly training may eventually result in a decrease in performance in the downstream tasks. The CURL model struggles to learn effective behavior policies, especially in Hopper Stand. Additional results on the challenging DMC Humanoid Walk can be found in Supplementary Material C.1. Additionally, we present qualitative results in Figure 5 and Figure 6. Figure 5 shows the traversals of $\beta$ -VAE during the pretraining phase. In each row of traversals, a distinct attribute varies, while other attributes remain constant, indicating that the pretrained model has successfully disentangled and learned this attribute, thereby improving the sample efficiency of the RL agent. Figure 6 displays the fine-grained disentanglement results on MuJoCo Pusher during the finetuning phase, demonstrating that the world model can effectively disentangle the variations.

# 4.3. Model Analyses

Ablation studies. We conduct ablation studies to validate the effect of the latent distillation and disentanglement constraints. Figure 7 (Left) shows corresponding results in the DMC Walker Walk Cheetah Run. The green curve shows that removing the latent distillation of DisWM results in a decreased performance, which indicates that the latent distillation is essential during the early training stage. For the model represented by the blue curve, we do not adopt disentanglement constraints for both pretrain and finetune stages. It can be seen that the necessity of introducing DRLbased training and disentangled representation significantly improves the learning efficiency of the agent.

![](images/3.jpg)  
Figure 3. Example image observations of our modified DMC and MuJoCo Pusher with color distractors.

![](images/4.jpg)  
Figure 4. Comparison of DisWM against visual RL baselines, including DreamerV2 [10], $A P V$ [29], DV2 Finetune, TED [7], CURL [18].

![](images/5.jpg)  
Figure 5. Visualization of traversals of $\beta$ VAE during the pretraining phase.

![](images/6.jpg)  
displays the traversal results on a specific attribute.

Sensitivity analyses. We conduct sensitivity analyses on DMC (Cheetah $R u n $ Walker Walk). As shown in Figure 7 (Middle), we observe that when $\beta$ for the representation disentanglement is too small, the model learns entangled latent representations. When $\beta$ is too large, it will impede the reconstruction of the image, leading to a decline in performance. Latent distillation weight $\eta$ controls the cross-domain transfer scale. Intuitively, setting this hyperparameter too low may result in the downstream agent not getting enough knowledge from the pretrained models. Conversely, excessively high $\eta$ may result in the model overfitting to the pretrained model, which is not conducive to the learning of the downstream task. Additional sensitivity analyses on the latent space dimension are provided in Supplementary Material C.3. Effects of video domain. In Figure 8, we evaluate DisWM on DMC Cartpole Swingup by pretraining on alternative video datasets, including frames collected from Finger Spin, Reacher Easy, Walker Walk, and Hopper Stand. Interestingly, compared with the DreamerV2 agent without pretraining, DisWM can always benefit from pretraining via offline-to-online latent distillation. It obtains the semantic knowledge from the pretrained model and strengthens the disentanglement capability during finetuning phase.

![](images/7.jpg)  
FgureThese gures llustrate theablation studies and sensitivity analyses o DisWMon DMC Walker Wal Cheetah Run. Left: cross-domain latent distillation weight. Right: The performance of DisWM with different disentanglement scale.

![](images/8.jpg)  
Figure 8. Performance of DisWM on DMC Cartpole Swingup with different video datasets.

# 5. Related Work

Visual MBRL. Visual RL learns control policies from raw pixels, which has achieved remarkable performance in various tasks [3, 4, 31, 37], while prior RL studies focus on learning policies from low-dimensional states. Extant approaches can be divided into two main directions: model-free RL [18, 19, 24, 32, 42, 44, 47] and model-based RL [1, 810, 12, 20, 21, 23, 27, 30, 36, 46]. The following methods specifically address the variations of environments in visual MBRL. Pan [28] et al. decompose visual dynamics into controllable and uncontrollable states through the optimization of inverse dynamics. SeeX [16] proposed a bilevel optimization framework that adopts a separated world model and maximizes the task-relevant uncertainty. Orthogonal to these studies, our approach employs a DRL-based world model to alleviate the issue of visual variations. Transfer RL To facilitate the learning of unseen tasks, transfer RL [22, 24, 26, 33, 36, 48] leverages the knowledge learned from past tasks. One promising way is to transfer world knowledge from accessible videos to improve the downstream control. APV [29] established a pretraining-finetuning framework with a stacked latent prediction model and video-based intrinsic bonus. IPV [39] introduced contextualized world models that pretrained on diverse in-the-wild videos. It incorporates a context encoder that works alongside the latent dynamics model into the image encoder to capture rich contextual information. PreLAR [45] pretrained the world model with the derived meaningful actions from the action-free video using an inverse dynamics encoder. Different from these approaches, we propose a new solution to transfer world knowledge from distracting videos to improve the learning efficiency of downstream tasks via offline-to-online latent distillation.

# 6. Conclusions and Limitations

In this paper, we present a transfer RL dubbed DisWM, which addresses the challenge of environment variations in practical scenarios. Our key insight is to leverage the accessible distracting videos to facilitate the sample efficiency of downstream tasks to offer flexible disentanglement constraints. Specifically, we introduce disentangled representation pretraining, offline-to-online latent distillation, and disentangled world model adaptation to improve the downstream control. DisWM demonstrates superior performance than existing visual RL baselines across various benchmarks. One limitation of our approach is that disentangled representation learning encounters challenges in complex environments. Exploring the non-stationary environments with more intricate variations, such as time-varying background video distractions, could further highlight the potential of our approach for practical scenarios.

Acknowledgements. This work was supported by Grants of NSFC 62302246 & 62250062, ZJNSFC LQ23F010008, Ningbo 2023Z237 & 2024Z284 & 2024Z289 & 2023CX050011 & 2025Z038, the Smart Grid National Science and Technology Major Project (2024ZD0801200), the Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102), the Fundamental Research Funds for the Central Universities, and the IDT Foundation of Youth Doctoral Innovation (S203.2.01.32.002). Additional support was provided by the project of Supporting Program for Young and Middle-aged Scientific and Technological Innovation Talents in Shenyang (Grant RC210488), the project of Provincial Doctoral Research Initiation Fund Program (Grant 2023-BS-214), the High Performance Computing Center at Eastern Institute of Technology, Ningbo, and Ningbo Institute of Digital Twin.

# References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. In NeurIPS, 2024. 8   
[2] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. TPAMI, 35(8):17981828, 2013. 2   
[3] Hyesong Choi, Hunsang Lee, Seongwon Jeong, and Dongbo Min. Environment agnostic representation for visual reinforcement learning. In ICCV, pages 263273, 2023. 8   
[4] Hyesong Choi, Hunsang Lee, Wonil Song, Sangryul Jeon, Kwanghoon Sohn, and Dongbo Min. Local-guided global: Paired similarity representation for visual reinforcement learning. In CVPR, pages 1507215082, 2023. 8   
[5] Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015. 1   
[6] Mhairi Dunion, Trevor McInroe, Kevin Sebastian Luck, Josiah Hanna, and Stefano Albrecht. Conditional mutual information for disentangled representations in reinforcement learning. In NeurIPS, 2023. 1, 2   
[7] Mhairi Dunion, Trevor McInroe, Kevin Sebastian Luck, Josiah P Hanna, and Stefano V Albrecht. Temporal disentanglement of representations for improved generalisation in reinforcement learning. In ICLR, 2023. 1, 2, 5, 6   
[8] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In ICML, 2019. 8   
[9] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In ICLR, 2020.   
10] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In ICLR, 2021. 4, 6, 8, 1   
[11] Danjar Harner, Kuang-Huel Lee, lan riscner, ana Peter Abbeel. Deep hierarchical planning from pixels. arXiv preprint arXiv:2206.04114, 2022. 1   
[12] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. Nature, 2025. 8, 1   
[13] Nicklas Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In ICLR, 2024. 1   
[14] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In ICLR, 2017. 2, 3   
[15] Irina Higgins, Arka Pal, Andrei Rusu, Loic Matthey, Christopher Burgess, Alexander Pritzel, Matthew Botvinick, Charles Blundell, and Alexander Lerchner. Darla: Improving zero-shot transfer in reinforcement learning. In ICML, 2017. 2   
[16] Kaichen Huang, Shenghua Wan, Minghao Shao, Hai-Hang Sun, Le Gan, Shuai Feng, and De-Chuan Zhan. Leveraging separated world model for exploration in visually distracted environments. NeurIPS, 2024. 8   
[17] Misha Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas. Reinforcement learning with augmented data. In NeurIPS, 2020. 5   
[18] Michael Laskin, Aravind Srinivas, and Pieter Abbeel. Curl: Contrastive unsupervised representations for reinforcement learning. In ICML, pages 56395650, 2020. 1, 6, 8   
[19] Haoran Li, Zhennan Jiang, YUHUI CHEN, and Dongbin Zhao. Generalizing consistency policy to visual rl with prioritized proximal experience regularization. In NeurIPS, 2024. 8   
[20] Jiajian Li, Qi Wang, Yunbo Wang, Xin Jin, Yang Li, Wenjun Zeng, and Xiaokang Yang. Open-world reinforcement learning over long short-term imagination. In ICLR, 2025. 1,8   
[21] Jessy Lin, Yuqing Du, Olivia Watkins, Danijar Hafner, Pieter Abbeel, Dan Klein, and Anca Dragan. Learning to model the world with language. In ICML, 2024. 8   
[22] Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob Foerster, Satinder Singh, and Feryal Behbahani. Structured state space models for in-context reinforcement learning. NeurIPS, 2023. 8   
[23] Haoyu Ma, Jialong Wu, Ningya Feng, Chenjun Xiao, Dong Li, Jianye Hao, Jianmin Wang, and Mingsheng Long. Harmonydream: Task harmonization inside world models. In ICML, 2024. 8   
[24] Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. In ICLR, 2023. 8   
[25] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. JMLR, 9(Nov):25792605, 2008. 2   
[26] Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Alexandre Lacoste, and Sai Rajeswar. Choreographer: Learning and adapting skills in imagination. In ICLR, 2023. 8   
[27] Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Aaron C Courville, and Sai Rajeswar Mudumba. Genrl: Multimodalfoundation world models for generalization in embodied agents. NeurIPS, 2024. 8   
[28] Minting Pan, Xiangming Zhu, Yunbo Wang, and Xiaokang Yang. Iso-dream: Isolating and leveraging noncontrollable visual dynamics in world models. In NeurIPS, 2022. 1, 8   
[29] Younggyo Seo, Kimin Lee, Stephen L James, and Pieter Abbeel. Reinforcement learning with action-free pretraining from videos. In ICML, 2022. 4, 5, 6, 8, 1   
[30] Younggyo Seo, Junsu Kim, Stephen James, Kimin Lee, Jinwoo Shin, and Pieter Abbeel. Multi-view masked world models for visual robotic manipulation. In ICML, 2023. 8   
[31] Wonil Song, Hyesong Choi, Kwanghoon Sohn, and Dongbo Min. A simple framework for generalization in visual rl under dynamic scene perturbations. NeurIPS, 37:121790 121826, 2024. 8   
[32] Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin. Decoupling representation learning from reinforcement learning. In ICML, pages 98709879, 2021. 8   
[33] Yanchao Sun, Ruijie Zheng, Xiyao Wang, Andrew Cohen, and Furong Huang. Transfer rl across observation feature spaces via model-based regularization. In ICLR, 2022. 8   
[34] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018. 4   
[35] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In IROS, 2012. 4   
[36] Qi Wang, Junming Yang, Yunbo Wang, Xin Jin, Wenjun Zeng, and Xiaokang Yang. Making offine rl online: Collaborative world models for offine visual reinforcement learning. In NeurIPS, 2024. 8   
[37] Xudong Wang, Long Lian, and Stella X Yu. Unsupervised visual attention and invariance for reinforcement learning. In CVPR, pages 66776687, 2021. 4, 8, 1   
[38] Xin Wang, Hong Chen, Zihao Wu, Wenwu Zhu, et al. Disentangled representation learning. TPAMI, 2024. 2   
[39] Jialong Wu, Haoyu Ma, Chaoyi Deng, and Mingsheng Long. Pre-training contextualized world models with in-the-wild videos for reinforcement learning. NeurIPS, 2023. 8, 1   
[40] Baao Xie, Bohan Li, Zequn Zhang, Junting Dong, Xin Jin, Jingyu Yang, and Wenjun Zeng. Navinerf: Nerf-based 3d representation disentanglement by latent semantic navigation. In ICCV, 2023. 2   
[41] Baao Xie, Qiuyu Chen, Yunnan Wang, Zequn Zhang, Xin Jin, and Wenjun Zeng. Graph-based unsupervised disentangled representation learning via multimodal large language models. In NeurIPS, 2024. 2   
[42] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Mastering visual continuous control: Improved dataaugmented reinforcement learning. In ICLR, 2022. 8   
[43] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Metaworld: A benchmark and evaluation for multi-task and meta reinforcement learning. In CoRL. 2019. 4   
[44] Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. In ICLR, 2021. 8   
[45] Lixuan Zhang, Meina Kan, Shiguang Shan, and Xilin Chen. Prelar: World model pre-training with learnable action representation. In ECCV, 2024. 1, 8   
[46] Weipu Zhang, Gang Wang, Jian Sun, Yetian Yuan, and Gao Huang. Storm: Efficient stochastic transformer based world models for reinforcement learning. In NeurIPS, 2023. 8   
[47] Ruijie Zheng, Xiyao Wang, Yanchao Sun, Shuang Ma, Jieyu Zhao, Huazhe Xu, Hal Daumé III, and Furong Huang. Taco: Temporal latent action-driven contrastive loss for visual reinforcement learning. In NeurIPS, 2023. 8   
[48] Zhuangdi Zhu, Kaixiang Lin, Anil K Jain, and Jiayu Zhou. Transfer learning in deep reinforcement learning: A survey. TPAMI, 45(11):1334413362, 2023. 8

# Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning

Supplementary Material

# A. Compared Baselines

We compare DisWM with strong visual RL agents, including • DreamerV2 [10]: A model-based RL (MBRL) approach that trains world model and learns by imagining future latent states. • APV [29]: It learns informational representations via action-free pretraining on videos and finetunes the agent with learned representations in the downstream tasks with action. •DV2 Finetune: It pretrains a DreamerV2 agent [10] on distracting videos and then finetunes the trained model in the downstream tasks. Note that some tasks have different action spaces, which makes it difficult to finetune directly. Therefore, the action space of two tasks is set as the maximum action space of both environments. •TED [7]: It adopts a classification task to learn temporally disentangled representations in visual RL. • CURL [18]: A model-free RL method that employs contrastive learning to improve its sample efficiency.

# B. Behavior Learning

For the behavior learning of DisWM, we adopt the actorcritic method following DreamerV2 [10]. Concretely, the actor and critic are both implemented as MLPs with ELU activations [5]. Formally, the actor and critic are defined as below:

$$
\begin{array} { r l } & { \mathrm { A c t o r : ~ } \hat { a } _ { t } \sim \pi _ { \psi } ( \hat { a } _ { t } | \hat { z } _ { t } ) } \\ & { \mathrm { C r i t i c : ~ } v _ { \xi } ( \hat { z } _ { t } ) \approx \mathbb { E } _ { p _ { \phi } , p _ { \psi } } \Big [ \sum _ { \tau \geq t } \hat { \gamma } _ { \tau - t } \hat { r } _ { \tau } \Big ] . } \end{array}
$$

The actor $\pi _ { \psi }$ is optimized by maximizing

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } ( \psi ) = \mathbb { E } _ { p _ { \phi } , p _ { \psi } } \Big [ \displaystyle \sum _ { t = 1 } ^ { H - 1 } ( \underbrace { \beta \mathrm { H } \left[ a _ { t } | \hat { z } _ { t } \right] } _ { \mathrm { e n t r o p y ~ r e g u l a r i z a t i o n } } + \underbrace { \rho V _ { t } } _ { \mathrm { d y n a m i c s ~ b a c k p r o p } } } \\ & { \displaystyle + \underbrace { ( 1 - \rho ) \ln \pi _ { \psi } ( \hat { a } _ { t } | \hat { z } _ { t } ) \mathrm { s g } ( V _ { t } - v _ { \xi } ( \hat { z } _ { t } ) ) } _ { \mathrm { R E I N F O R C E } } \Big ] . } \end{array}
$$

We train the critic $v _ { \xi }$ by minimizing where $\mathtt { S g }$ is a stop gradient operator.

$$
\mathcal { L } ( \xi ) = \mathbb { E } _ { p _ { \phi } , p _ { \psi } } \Big [ \sum _ { t = 1 } ^ { H - 1 } \frac { 1 } { 2 } \left( v _ { \xi } \left( \hat { z } _ { t } \right) - \mathrm { s g } \left( V _ { t } \right) \right) ^ { 2 } \Big ] .
$$

The $\lambda$ -target $V _ { t }$ that involves a weighted average of reward information used in Eq. (7) and Eq. (8) is defined as:

$$
V _ { t } \doteq { \hat { r } } _ { t } + { \hat { \gamma } } _ { t } \left\{ { \begin{array} { l l } { ( 1 - \lambda ) v _ { \xi } \left( { \hat { z } } _ { t + 1 } \right) + \lambda V _ { t + 1 } } & { { \mathrm { i f ~ } } t < H } \\ { v _ { \xi } \left( { \hat { z } } _ { H } \right) } & { { \mathrm { i f ~ } } t = H } \end{array} } \right. .
$$

where $H$ is the imagination horizon. Notably, the disentangled world model is not optimized during behavior learning.

# C. Additional Results

# C.1. Results on DMC

We compare the performance of DreamerV3 [12], TDMPC2 [13], ContextWM [39], and our approach on DMC. As shown Table A, DisWM outperforms other strong baselines in terms of episode return.

# C.2. Results on DrawerWorld

We present results on DrawerWorld [37] in Table B. As reported in Table B, DisWM (source: Finger Spin) outperforms other baselines in terms of success rate $( \% )$ on all tasks.

# C.3. Sensitivity of the Latent Space Dimension

We visualize sensitivity analyses on the latent space dimension in Figure I. We observe that when $\mathbf { z } _ { \mathrm { d i m } }$ for the $\beta$ VAE is too small, it impedes the learning of disentangled representations, leading to a decline in performance.

![](images/9.jpg)  
Figure I. Sensitivity analyses on Cheetah Run Walker Walk

Table A. Comparison with strong baselines on DMC.

<table><tr><td>Model</td><td></td><td>Reacher Easy → Cheetah Run Walker Walk → Humanoid Walk</td></tr><tr><td>DreamerV3</td><td>662 ± 9</td><td>12 ± 17</td></tr><tr><td>TD-MPC2</td><td>510 ± 15</td><td>1 ± 0</td></tr><tr><td>ContextWM</td><td>661 ± 49</td><td>1 ± 0</td></tr><tr><td>DisWM</td><td>817 ± 59</td><td>147 ± 85</td></tr></table>

<table><tr><td>Model</td><td>DrawerClose</td><td>DrawerOpen</td></tr><tr><td>TDMPC2</td><td>3 ± 6</td><td>43 ± 25</td></tr><tr><td>ContextWM</td><td>37 ± 12</td><td>23 ± 25</td></tr><tr><td>DisWM</td><td>77 ± 6</td><td>70 ± 10</td></tr></table>

Table B. Performance on DrawerWorld with texture variations.

# C.4. Runtime Comparisons

We provide the detailed runtime and parameter comparisons with baselines in Table C. Note that the inference time is computed for one episode. Table C. Runtime and model size comparisons evaluated on DMC (Finger Spin Reacher Easy). DV2 FT is short for DreamerV2 finetune.   

<table><tr><td>Model</td><td>Training Steps</td><td>Training time</td><td>Inference time</td><td>Params (M)</td></tr><tr><td>CURL</td><td>100k</td><td>303 min</td><td>4.97 sec</td><td>10.7</td></tr><tr><td>DV2 FT</td><td>200k</td><td>1522 min</td><td>9.88 sec</td><td>12.1</td></tr><tr><td>APV</td><td>200k</td><td>1722 min</td><td>10.15 sec</td><td>13</td></tr><tr><td>TED</td><td>100k</td><td>1051 min</td><td>20.49 sec</td><td>11.5</td></tr><tr><td>DV2</td><td>100k</td><td>901 min</td><td>9.59 sec</td><td>12.1</td></tr><tr><td>DisWM</td><td>200k</td><td>1311 min</td><td>9.48 sec</td><td>5.8</td></tr></table>

# C.5. Sample Diversity Visualization

The adaptation stage enriches the sample diversity, as shown in Figure J, for Cheetah $R u n $ Walker Walk, we sample 200 video clips of length 50 and visualize the corresponding latent features using t-SNE [25]. We find that the latent features of the online interactions are more diverse than those of the offline dataset.

# D. Hyperparameters

The final hyperparameters of DisWM are reported in Table D.

![](images/10.jpg)  
Figure J. Sample diversity enhanced by adaptation.

Table D. Hyperparameters of DisWM.

<table><tr><td>Name</td><td>Notation</td><td>Value</td></tr><tr><td>Video prediction model</td><td></td><td></td></tr><tr><td>Image size KL divergence scale</td><td>β1</td><td>64 × 64 1</td></tr><tr><td>Disentanglement scale Latent dimension</td><td>β2</td><td>0.015 20</td></tr><tr><td>Learning rate</td><td></td><td>3 ·10−4</td></tr><tr><td>Disentangled World Model</td><td></td><td></td></tr><tr><td>Latent distillation weight</td><td>η</td><td>0.1</td></tr><tr><td>Disentanglement scale</td><td>β</td><td>0.015</td></tr><tr><td>KL divergence scale</td><td>α</td><td>1</td></tr><tr><td>Latent dimension</td><td>−</td><td>20</td></tr><tr><td>Batch size</td><td>B</td><td>50</td></tr><tr><td>Batch length</td><td>L</td><td>50</td></tr><tr><td>Learning rate</td><td></td><td>3 · 10−4</td></tr><tr><td>Behavior Learning</td><td></td><td></td></tr><tr><td>Imagination horizon</td><td>H</td><td>15</td></tr><tr><td>Discount</td><td>γ</td><td>0.99</td></tr><tr><td>λ-target</td><td>λ</td><td>0.95</td></tr><tr><td>Actor learning rate</td><td></td><td>8·10-5</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Critic learning rate</td><td></td><td>8·10-5</td></tr></table>