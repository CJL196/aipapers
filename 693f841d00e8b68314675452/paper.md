# From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

Tianwei Yin1\* Qiang Zhang2\* Richard Zhang2 William T. Freeman1 Frédo Durand1 Eli Shechtman2 Xun Huang2 1MIT 2 Adobe https://causvid.github.io/

![](images/1.jpg)  
at approximately 9.4 FPS, facilitating interactive workflows for video content creation.

# Abstract

Current video diffusion models achieve impressive generation quality but struggle in interactive applications due to bidirectional attention dependencies. The generation of a single frame requires the model to process the entire sequence, including the future. We address this limitation by adapting a pretrained bidirectional diffusion transformer to an autoregressive transformer that generates frames on-thefly. To further reduce latency, we extend distribution matching distillation (DMD) to videos, distilling 50-step diffusion model into a 4-step generator. To enable stable and high-quality distillation, we introduce a student initialization scheme based on teacher's ODE trajectories, as well as an asymmetric distillation strategy that supervises a causal student model with a bidirectional teacher. This approach effectively mitigates error accumulation in autoregressive generation, allowing long-duration video synthesis despite training on short clips. Our model achieves a total score of 84.27 on the VBench-Long benchmark, surpassing all previous video generation models. It enables fast streaming generation of high-quality videos at 9.4 FPS on a single GPU thanks to KV caching. Our approach also enables streaming video-to-video translation, image-to-video, and dynamic prompting in a zero-shot manner. We release our code and pretrained models.

# 1. Introduction

The emergence of diffusion models has revolutionized how we can create videos from text [3, 5, 24, 28, 61, 96, 109]. Many of the state-of-the-art video diffusion models rely on the Diffusion Transformer (DiT) architecture [2, 60], which usually employs bidirectional attention across all video frames. Despite the impressive quality, the bidirec

![](images/2.jpg)

![](images/3.jpg)  
"A little boy jumping in the air over a puddle of water."

![](images/4.jpg)  
A photorealistic video of a yellow sports car driving down a road, with trees in the background."

# Dynamic Prompting tional dependencies imply that generating a single frame requires processing the entire video. This introduces long latency and prevents the model from being applied to interactive and streaming applications, where the model needs to continually generate frames based on user inputs that may change over time. The generation of the current frame depends on future conditional inputs that are not yet available. Current video diffusion models are also limited by their speed. The compute and memory costs increase quadratically with the number of frames, which, combined with the large number of denoising steps during inference, makes generating long videos prohibitively slow and expensive.

![](images/5.jpg)  
"A woman is walking on the street, towards the camera"   
"[] She adjusts the collar of her trench coat and tilts her fedora [...] She pulls out a notebook, making a quick note[...] street lamps that create pools of yellow light"   
"[….] in an off-shoulder white dress [.…] She pauses to adjust her sunglasses[...] and then snaps a quick photo with her phone[…] "   
a video to build extended narratives with evolving actions and environments.

Autoregressive models offer a promising solution to address some of these limitations, but they face challenges with error accumulation and computational efficiency. Instead of generating all frames simultaneously, autoregressive video models generate frames sequentially. Users can start watching the video as soon as the first frame is generated, without waiting for the entire video to be completed. This reduces latency, removes limitations on video duration, and opens the door for interactive control. However, autoregressive models are prone to error accumulation: each generated frame builds on potentially flawed previous frames, causing prediction errors to magnify and worsen over time. Moreover, although the latency is reduced, existing autoregressive video models are still far from being able to generate realistic videos at interactive frame rate [7, 28, 36].

In this paper, we introduce CausVid, a model designed for fast and interactive causal video generation. We design an autoregressive diffusion transformer architecture with causal dependencies between video frames. Similar to the popular decoder-only large language models (LLMs) [6, 64], our model achieves sample-efficient training by leveraging supervision from all input frames at each iteration, as well as efficient autoregressive inference through key-value (KV) caching. To further improve generation speed, we adapt distribution matching distillation (DMD) [99, 100], a few-step distillation approach originally designed for image diffusion models, to video data. Instead of naively distilling an autoregressive diffusion model [8, 28] into a few-step student, we propose an asymmetric distillation strategy where we distill the knowledge in a pretrained teacher diffusion model with bidirectional attention into our causal student model. We show that this asymmetric distillation approach significantly reduced error accumulation during autoregressive inference. This allows us to support autoregressively generating videos that are much longer than the ones seen during training. Comprehensive experiments demonstrate that our model achieves video quality on par with state-of-the-art bidirectional diffusion models while offering enhanced interactivity and speed. To our knowledge, this is the first autoregressive video generation method that competes with bidirectional diffusion in terms of quality (Appendix Fig. 3 and Fig. 4). Additionally, we showcase the versatility of our method in tasks such as image-to-video generation, video-to-video translation, and dynamic prompting, all achievable with extremely low latency (Fig. 2).

# 2. Related Work

Autoregressive Video Generation. Given the inherent temporal order of video data, it is intuitively appealing to model video generation as an autoregressive process. Early research uses either regression loss [19, 48] or GAN loss [37, 57, 78, 81] to supervise the frame prediction task. Inspired by the success of LLMs [6], some works choose to tokenize video frames into discrete tokens and apply autoregressive transformers to generate tokens one by one [17, 36, 42, 86, 90, 95]. However, this approach is computationally expensive as each frame usually consists of thousands of tokens. Recently, diffusion models have emerged as a promising approach for video generation. While most video diffusion models have bidirectional dependencies [5, 61, 96, 109], autoregressive video generation using diffusion models has also been explored. Some works [1, 28, 79, 104] train video diffusion models to denoise new frames given context frames. Others [8, 33, 68] train the model to denoise the entire video under the setting where different frames may have different noise levels. Therefore, they support autoregressive sampling as a special case where the current frame is noisier than previous ones. A number of works have explored adapting pretrained textto-image [35, 41, 79, 88] or text-to-video [16, 22, 33, 91, 93] diffusion models to be conditioned on context frames, enabling autoregressive video generation. Our method is closely related to this line of work, with the difference that we introduce a novel adaption method through diffusion distillation, significantly improving efficiency and making autoregressive methods competitive with bidirectional diffusion for video generation. Long Video Generation. Generating long and variablelength videos remains a challenging task. Some works [11, 63, 77, 82, 83, 102, 106] generate multiple overlapped clips simultaneously using video diffusion model pretrained on fixed and limited-length clips, while employing various techniques to ensure temporal coherence. Another approach is to generate long videos hierarchically, first generating sparse keyframes and then interpolating between them [98, 107]. Unlike full-video diffusion models that are trained to generate videos of fixed length, autoregressive models [16, 22, 28, 42, 86, 91] are inherently suitable for generating videos of various length, although they may suffer from error accumulation when generating long sequences. We find that the distribution matching objective with a bidirectional teacher is surprisingly helpful for reducing the accumulation of errors, enabling both efficient and high-quality long video generation.

Diffusion Distillation. Diffusion models typically require many denoising steps to generate high-quality samples, which can be computationally expensive [23, 73]. Distillation techniques train a student model to generate samples in fewer steps by mimicking the behavior of a teacher diffusion model [29, 52, 59, 65, 69, 70, 76, 94, 100]. Luhman et al. [52] train a single-step student network to approximate the noise-image mapping obtained from a DDIM teacher model [73]. Progressive Distillation [69] trains a sequence of student models, reducing the number of steps by half at each stage. Consistency Distillation [21, 32, 53, 74, 76] trains the student to map any points on an ODE trajectory to its origin. Rectified flow [14, 46, 47] trains a student model on the linear interpolation path of noise-image pairs obtained from the teacher. Adversarial loss [18] is also used, sometimes in combination with other methods, to improve the quality of student output [29, 44, 70, 94, 99]. DMD [99, 100] optimizes an approximate reverse KL divergence [15, 54, 87, 97], whose gradients can be represented as the difference of two score functions trained on the data and generator's output distribution, respectively. Unlike trajectory-preserving methods [32, 47, 69], DMD provides supervision at the distribution level and offers the unique advantage of allowing different architectural formulations for the teacher and student diffusion models. Our approach builds upon the effectiveness and flexibility of DMD to train an autoregressive generator by distilling from a bidirectional teacher diffusion model. Recently, researchers have begun to apply distillation methods to video diffusion models, such as progressive dis

![](images/6.jpg)  
o the wind as they walk [.]"

![](images/7.jpg)  
damotnhot  paplane orhi int swanThe pont ose bees  au  d head, wings unfolding and expanding [...]"

![](images/8.jpg)  
e a temperature warms."

![](images/9.jpg)  
illustrating the power and destruction of such an event."

![](images/10.jpg)  
er gorgeous and soft and sun-kissed, with golden backlight and dreamy bokeh and lens flares [..]"

![](images/11.jpg)  
w vast grassy hills lie in the distant background […]"

![](images/12.jpg)  
okantmansinwit frindly jckantens n hostharacter welomitrick treater e entrance, tilt shift photography."

F orders of magnitude speedup. Please visit our website for more visualizations.

![](images/13.jpg)  
c

![](images/14.jpg)  
Young woman watching virtual reality in VR glasses in her living room."

![](images/15.jpg)  
close up portrait of young bearded guy with long beard."

![](images/16.jpg)  
.

![](images/17.jpg)

![](images/18.jpg)  

outputs. Please visit our website for more visualizations. tillation [43], consistency distillation [39, 56, 84, 85], and adversarial distillation [56, 105]. Most approaches focus on distilling models designed to generate short videos (less than 2 seconds). Moreover, they focus on distilling a noncausal teacher into a student that is also non-causal. In contrast, our method distills a non-causal teacher into a causal student, enabling streaming video generation. Our generator is trained on 10-second videos and can generate infinitely long videos via sliding window inference. There has been another line of work that focuses on improving the efficiency of video diffusion models by system-level optimization (e.g., caching and parallelism) [45, 103, 108, 112]. However, they are usually applied to standard multi-step diffusion models and can be combined with our distillation approach, further improving the throughput and latency.

# 3. Background

This section provides background information on video diffusion models (Sec. 3.1) and distribution matching distillation (Sec. 3.2), which our method is built upon.

# 3.1. Video Diffusion Models

Diffusion models [23, 72] generate samples from a data distribution $p ( x _ { 0 } )$ by progressively denoising samples that are initially drawn from a Gaussian distribution $p ( x _ { T } )$ . They are trained to denoise samples created by adding random noise $\epsilon$ to the samples $x _ { 0 }$ from the data distribution where $\alpha _ { t } , \sigma _ { T } > 0$ are scalars that jointly define the signalto-noise ratio according to a specific noise schedule [30, 34, 75] at step $t$ . The denoiser with parameter $\theta$ is typically trained to predict the noise [23]

$$
\begin{array} { r } { x _ { t } = \alpha _ { t } x _ { 0 } + \sigma _ { t } \epsilon , \epsilon \sim { \mathcal N } ( 0 , I ) , } \end{array}
$$

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathbb { E } _ { t , x _ { 0 } , \epsilon } \left. \epsilon _ { \theta } ( x _ { t } , t ) - \epsilon \right. _ { 2 } ^ { 2 } . } \end{array}
$$

Alternative prediction targets include the clean image $x _ { 0 }$ [30, 69] or a weighted combination of $x _ { 0 }$ and $\epsilon$ known as v-prediction [69]. All prediction schemes are fundamentally related to the score function, which represents the gradient of the log probability of the distribution [34, 75]:

$$
s _ { \theta } ( x _ { t } , t ) = \nabla _ { x _ { t } } \log p ( x _ { t } ) = - \frac { \epsilon _ { \theta } ( x _ { t } , t ) } { \sigma _ { t } } .
$$

In the following sections, we simplify our notation by using the score function $s _ { \theta }$ as a general representation of the diffusion model, while noting that it can be derived through reparameterization from a pretrained model of any prediction scheme. At inference time, we start from full Gaussian noise $x _ { T }$ and progressively apply the diffusion model to generate a sequence of increasingly cleaner samples. There are many possible sampling methods [30, 51, 73, 101] to compute the sample $x _ { t - 1 }$ in the next time step from the current one $x _ { t }$ based on the predicted noise $\epsilon _ { \theta } ( x _ { t } , t )$ . Diffusion models can be trained on either raw data [23, 27, 30] or on a lower-dimensional latent space obtained by a variational autoencoder (VAE) [31, 60, 66, 96, 109]. The latter is often referred to as latent diffusion models (LDMs) and has become the standard approach for modeling highdimensional data such as videos [4, 25, 96, 109, 111]. The autoencoder usually compresses both spatial and temporal dimensions of the video, making diffusion models easier to learn. The denoiser network in video diffusion models can be instantiated by different neural architectures, such as UNet [10, 24, 67, 111] or Transformers [5, 25, 80, 96].

# 3.2. Distribution Matching Distillation

Distribution matching distillation is a technique designed to distill a slow, multi-step teacher diffusion model into an efficient few-step student model [99, 100]. The core idea is to minimize the reverse KL divergence across randomly sampled timesteps $t$ between the smoothed data distribution $p _ { \mathrm { d a t a } } ( x _ { t } )$ and the student generator's output distribution $p _ { \mathrm { g e n } } ( x _ { t } )$ . The gradient of the reverse KL can be approximated as the difference between two score functions:

$$
\begin{array} { r l r } {  { \nabla _ { \phi } \mathcal { L } _ { \mathrm { D M D } } \triangleq \mathbb { E } _ { t } ( \nabla _ { \phi } \mathrm { K L } ( p _ { \mathrm { g e n } , t } \Vert p _ { \mathrm { d a t a } , t } ) ) } } \\ & { } & { \approx - \mathbb { E } _ { t } ( \int ( s _ { \mathrm { d a t a } } ( \Psi ( G _ { \phi } ( \epsilon ) , t ) , t ) \ d t ) \ d t ) } \\ & { } & { \qquad - s _ { \mathrm { g e n } , \xi } ( \Psi ( G _ { \phi } ( \epsilon ) , t ) , t ) ) \frac { d G _ { \phi } ( \epsilon ) } { d \phi } d \epsilon ) , } \end{array}
$$

where $\Psi$ represents the forward diffusion process as defined in Eq. 1, $\epsilon$ is random Gaussian noise, $G _ { \phi }$ is the generator parameterized by $\phi$ , and $s _ { \mathrm { d a t a } }$ and $s _ { \mathrm { g e n } , \xi }$ represent the score functions trained on the data and generator's output distribution, respectively, using a denoising loss (Eq. 2). During training, DMD [100] initializes both score functions from a pre-trained diffusion model. The score function of the data distribution is frozen, while the score function of the generator distribution is trained online using the generator's outputs. Simultaneously, the generator receives gradients to align its output with the data distribution (Eq. 4). DMD2 [99] extends this framework from single-step to multi-step generation by replacing the pure random noise input $\epsilon$ with a partially denoised intermediate image $x _ { t }$ .

# 4. Methods

Our approach introduces an autoregressive diffusion transformer that enables sequential video generation (Sec. 4.1). We show our training procedure in Fig. 6, which uses asymmetric distillation (Sec. 4.2) and ODE initialization (Sec. 4.3) to achieve high generation quality and stable convergence. We achieve efficient streaming inference through KV caching mechanisms (Sec. 4.4).

# 4.1. Autoregressive Architecture

We begin by compressing the video into a latent space using a 3D VAE. The VAE encoder processes each chunk of video frames independently, compressing them into shorter chunks of latent frames. The decoder then reconstructs the original video frames from each latent chunk. Our causal diffusion transformer operates in this latent space, generating latent frames sequentially. We design a blockwise causal attention mechanism inspired by prior works that combine autoregressive models with diffusion [38, 40, 49, 110]. Within each chunk, we apply bidirectional selfattention among latent frames to capture local temporal dependencies and maintain consistency. To enforce causality, we apply causal attention across chunks. This prevents frames in the current chunk from attending to frames in future chunks. A visual illustration of the architecture of our autoregressive diffusion transformer is provided in Fig. 5. Our design maintains the same latency as fully causal attention, as the VAE decoder still requires at least a block of latent frames to generate pixels. Formally, we define the attention mask $M$ as

![](images/19.jpg)  
and previous frames, but not the future.

$$
M _ { i , j } = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \left\lfloor { \frac { j } { k } } \right\rfloor \le \left\lfloor { \frac { i } { k } } \right\rfloor , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.
$$

Here, $i$ and $j$ index the frames in the sequence, $k$ is the chunk size, and $\lfloor \cdot \rfloor$ denotes the floor function. Our diffusion model $G _ { \phi }$ extends the DiT architecture [60] for autoregressive video generation. We introduce block-wise causal attention masks to the self-attention layers (illustrated in Fig. 6) while preserving the core structure, allowing us to leverage pretrained bidirectional weights for faster convergence.

# 4.2. Bidirectional Causal Generator Distillation

A straightforward approach to training a few-step causal generator would be through distillation from a causal

# Algorithm 1 Asymmetric Distillation with DMD

Require: Few-step denoising timesteps $\mathcal { T } = \{ 0 , t _ { 1 } , t _ { 2 } , \ldots , t _ { Q } \}$ , video length $N$ , chunk size $k$ , pretrained bidirectional teacher model $s \mathrm { d a t a }$ , dataset $\mathcal { D }$ .   
1: Initialize student model $G _ { \phi }$ with ODE regression (Sec. 4.3)   
2: Initialize generator output's score function $s _ { \mathrm { g e n } , \xi }$ with $S _ { \mathrm { { d a t a } } }$   
3: while training do   
4: Sample a video from the dataset and divide frames into $L = \lceil N / k \rceil$ chunks, $\{ x _ { 0 } ^ { i } \} _ { i = 1 } ^ { L } \sim \mathcal { D }$ .   
5: Sample per-chunk timesteps $\{ t ^ { i } \} _ { i = 1 } ^ { L } \sim \mathrm { U n i f o r m } ( \mathcal { T } )$   
6: Add noise: $x _ { t ^ { i } } ^ { i } = \alpha _ { t ^ { i } } x _ { 0 } ^ { i } + \sigma _ { t ^ { i } } \epsilon ^ { \dot { i } }$ , $\epsilon ^ { i } \sim \mathcal { N } ( 0 , I )$   
7: Predict clean frames with the student (block-wise causal mask): $\hat { x } _ { 0 } = G _ { \phi } \Big ( \{ x _ { t ^ { i } } ^ { i } \} , \{ t ^ { i } \} \Big )$   
8: Sample a single timestep $\dot { t } \sim \mathrm { U n i f o r m } ( 0 , T )$   
9: Add noise to predictions: $\hat { x } _ { t } = \alpha _ { t } \hat { x } _ { 0 } + \sigma _ { t } \epsilon , \epsilon \sim \mathcal { N } ( 0 , I )$   
10: Update student model $G _ { \phi }$ using DMD loss ${ \mathcal { L } } _ { \mathrm { D M D } } \triangleright$ Eq. 4   
11: Train generator's score function $s _ { \mathrm { g e n } , \xi }$ .   
12: Sample new noise $\epsilon ^ { \prime } \sim \mathcal { N } ( 0 , I )$   
13: Generate noisy $\hat { x } _ { t }$ : $\hat { x } _ { t } = \alpha _ { t } \hat { x } _ { 0 } + \sigma _ { t } \epsilon ^ { \prime }$   
14: Compute denoising loss and update $s _ { \mathrm { g e n } , \xi }$ . Eq. 2   
15: end while teacher model. This involves adapting a pretrained bidirectional DiT model by incorporating causal attention mechanism described above and fine-tuning it using the denoising loss (Eq. 2). During training, the model takes as input a sequence of $N$ noisy video frames divided into $L$ chunks $\{ x _ { t } ^ { i } \} _ { i = 1 } ^ { L }$ , where $i \in \{ 1 , 2 , . . . , L \}$ denotes the chunk index. Each chunk $\boldsymbol { x } _ { t } ^ { i }$ has its own noise time step $t ^ { i } \sim [ 0 , 9 9 9 ]$ , following Diffusion Forcing [8]. During inference, the model denoises each chunk sequentially, conditioned on the previously generated clean chunks of frames. While distilling this fine-tuned autoregressive diffusion teacher appears promising in theory, our initial experiments indicated that this naive approach yields suboptimal results. Since causal diffusion models typically underperform their bidirectional counterparts, training a student model from a weaker causal teacher inherently limits the student's capabilities. Moreover, issues such as error accumulation would propagate from teacher to student. To overcome the limitations of a causal teacher, we propose an asymmetric distillation approach: following state-of-the-art video models [5, 61], we employ bidirectional attention in the teacher model while constraining the student model to causal attention (Fig. 6 bottom). Algorithm 1 details our training process.

![](images/20.jpg)  
Figure 6.Our method distill a many-step, bidirectional videodiffusion model $s \mathrm { d a t a }$ into a 4-step, causal generator $G _ { \phi }$ The training

# 4.3. Student Initialization

Directly training the causal student model using the DMD loss can be unstable due to architectural differences. To address this, we introduce an efficient initialization strategy to stabilize training (Fig. 6 top). We create a small dataset of ODE solution pairs generated by the bidirectional teacher model: Sample a sequence of noise inputs $\{ x _ { T } ^ { i } \} _ { i = 1 } ^ { L }$ from the standard Gaussian distribution $\mathcal { L } ( 0 , I )$ : •Simulate the reverse diffusion process with an ordinary differential equation (ODE) solver [73] using the pretrained bidirectional teacher model to obtain the corresponding ODE trajectory $\{ x _ { t } ^ { i } \} _ { i = 1 } ^ { L }$ ,where $t$ spans $T$ to $0$ covering all inference timesteps. From the ODE trajectories, we select a subset of $t$ values that match those used in our student generator. The student model is then trained on this dataset with a regression loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { i n i t } } = \mathbb { E } _ { \boldsymbol { x } , t ^ { i } } | | G _ { \phi } ( \{ x _ { t ^ { i } } ^ { i } \} _ { i = 1 } ^ { N } , \{ t ^ { i } \} _ { i = 1 } ^ { N } ) - \{ x _ { 0 } ^ { i } \} _ { i = 1 } ^ { N } | | ^ { 2 } , } \end{array}
$$

where the few-step generator $G _ { \phi }$ is initialized from the teacher model. Our ODE initialization is computationally efficient, requiring only a small number of training iterations on relatively few ODE solution pairs.

# 4.4. Efficient Inference with KV Caching

During inference, we generate video frames sequentially using our autoregressive diffusion transformer with KV caching for efficient computation [6]. We show the detailed inference procedure in Algorithm 2. Notably, because we employ KV caching, block-wise causal attention is no longer needed at inference time. This allows us to leverage a fast bidirectional attention implementation [12].

# 5. Experiments

Models. Our teacher model is a bidirectional DiT [60] with an architecture similar to CogVideoX [96]. The model is trained on the latent space produced by a 3D VAE that encodes 16 video frames into a latent chunk consisting of 5 latent frames. The model is trained on 10-second videos at

# Algorithm 2 Inference Procedure with KV Caching

Require: Denoising timesteps $\{ t _ { 0 } = 0 , t _ { 1 } , \ldots , t _ { Q } \}$ , video length $N$ , chunk size $k$ , few-step autoregressive video generator $G _ { \phi }$ 1Divide frames into $L = \lceil N / k \rceil$ chunks 2: Initialize KV cache $\mathbf C \gets \emptyset$ 3: for $i = 1$ to $L$ do   
4: Initialize current chunk: xtQ $x _ { t _ { Q } } ^ { i } \sim \mathcal { N } ( 0 , I )$   
5: Iterative denoising over timesteps:   
6: for $j = Q$ to 1 do   
7: Generate output: $\hat { x } _ { t _ { j } } ^ { i } = G _ { \phi } ( x _ { t _ { j } } ^ { i } , t _ { j } )$ using cache C   
8: Update chunk: $x _ { t _ { j - 1 } } ^ { i } = \alpha _ { t _ { j - 1 } } \hat { x } _ { t _ { j } } ^ { i } + \sigma _ { t _ { j - 1 } } \epsilon ^ { \prime }$   
9: end for   
10: Update KV cache:   
11: Compute KV pairs with a forward pass $G _ { \phi } ( x _ { 0 } ^ { i } , 0 )$   
12: Append new KV pairs to cache C   
13:end for   
14: Return $\{ x _ { 0 } ^ { i } \} _ { i = 1 } ^ { L }$ a resolution of $3 5 2 \times 6 4 0$ and $1 2 \mathrm { F P S }$ .Our student model has the same architecture as the teacher, except that it employs causal attention where each token can only attend to other tokens within the same chunk and in preceding chunks. Each chunk contains 5 latent frames. During inference, it generates one chunk at a time using 4 denoising steps, with inference timesteps uniformly sampled as [999, 748, 502, 247]. We use FlexAttention [20] for efficient attention computation during training.

Training. We distill our causal student model using a mixed set of image and video datasets following CogVideoX [96]. Images and videos are filtered based on safety and aesthetic scores [71]. All videos are resized and cropped to the training resolution $( 3 5 2 \times 6 4 0 )$ and we use around 400K singleshot videos from an internal dataset, for which we have full copyright. During training, we first generate 1000 ODE pairs (Sec. 4.3) and train the student model for 3000 iterations with AdamW [50] optimizer and a learning rate of $5 \times 1 0 ^ { - 6 }$ . After that, we train with our asymmetric DMD loss (Sec. 4.2) with the AdamW optimizer and a learning rate of $2 \times 1 0 ^ { - 6 }$ for 6000 iterations. We use a guidance scale of 3.5 and adopt the two time-scale update rule from DMD2 [99] with a ratio of 5. The whole training process takes around 2 days on 64 H100 GPUs. Evaluation. Our method is evaluated on VBench [26], a benchmark for video generation with 16 metrics designed to systematically assess motion quality and semantic alignment. For our main results, we use the first 128 prompts from MovieGen [61] to generate videos and evaluate model performance on three primary aspects from VBench competition's evaluation suite. A comprehensive evaluation using all prompts from VBench is provided in the appendix. The inference times are measured on a H100 GPU.

![](images/21.jpg)  
Figure 7. User study comparing our distilled causal video generator with its teacher model and existing video diffusion models. Our model demonstrates superior video quality (scores $> 5 0 \%$ , while achieving a significant reduction in latency by multiple orders of magnitude.

# 5.1. Text to Video Generation

We evaluate the ability of our approach to generate short videos (5 to 10 seconds) and compare it against state-ofthe-art methods: CogVideoX [96], OpenSORA [109], Pyramid Flow [28], and MovieGen [61]. As shown in Tab. 1, our method outperforms all baselines across all three key metrics: temporal quality, frame quality, and text alignment. Our model achieves the highest temporal quality score of 94.7, indicating superior motion consistency and dynamic quality. In addition, our method shows notable improvements in frame quality and text alignment, scoring 64.4 and 30.1, respectively. In the supplementary material, we present our method's performance on the VBench-Long leaderboard, achieving a total score of 84.27 and securing first place among all officially evaluated video generation models. We further evaluate our model's performance through a human preference study. We select the first 29 prompts from the MovieGenBench dataset and collect ratings from independent evaluators using the Prolific platform. For each pair of compared models and each prompt, we collect 3 ratings from different evaluators, resulting in a total of 87 ratings per model pair. The evaluators choose the better video between the two generated videos based on the visual quality and semantic alignment with the input prompt. The specific questions and interface are shown in Appendix Fig. 11. For reproducibility, we use a fixed random seed of zero for all videos. As illustrated in Fig. 7, our model consistently outperforms baseline methods such as MovieGen, CogVideoX, and Pyramid Flow. Notably, our distilled model maintains performance comparable to the bidirectional teacher while offering orders of magnitude faster inference, validating the effectiveness of our approach. We also compare our method with prior works designed for long video generation: Gen-L-Video [82], FreeNoise [63], StreamingT2V [22], FIFO-Diffusion [33], and Pyramid Flow [28]. We use a sliding window inference strategy, taking the final frames of the previous 10- second segment as context for generating the next segment. The same strategy is also applied to generate long videos using Pyramid Flow. Tab. 2 shows that our method outperforms all baselines in terms of temporal quality and frame-wise quality and is competitive in text alignment. It also successfully prevents error accumulation. As shown in Fig. 8, our method maintains image quality over time, while most autoregressive baselines suffer from quality degradation [8, 22, 28]. Tab. 3 compares the efficiency of our method with competing approaches [28, 96] and our bidirectional teacher diffusion model. Our method achieves a $1 6 0 \times$ reduction in latency and a $1 6 \times$ improvement in throughput compared to the similarly scaled CogVideoX [96].

Table 1. Evaluation of text-to-short-video generation. Each method is evaluated at its closest supported length to 10 seconds.   

<table><tr><td>Method</td><td>Length (s)</td><td>Temporal Quality</td><td>Frame Quality</td><td>Text Alignment</td></tr><tr><td>CogVideoX-5B</td><td>6</td><td>89.9</td><td>59.8</td><td>29.1</td></tr><tr><td>OpenSORA</td><td>8</td><td>88.4</td><td>52.0</td><td>28.4</td></tr><tr><td>Pyramid Flow</td><td>10</td><td>89.6</td><td>55.9</td><td>27.1</td></tr><tr><td>MovieGen</td><td>10</td><td>91.5</td><td>61.1</td><td>28.8</td></tr><tr><td>CausVid (Ours)</td><td>10</td><td>94.7</td><td>64.4</td><td>30.1</td></tr></table>

Table 2. Evaluation of text-to-long-video generation. All methods produce videos approximately 30s in length.   

<table><tr><td>Method</td><td>Temporal Quality</td><td>Frame Quality</td><td>Text Alignment</td></tr><tr><td>Gen-L-Video</td><td>86.7</td><td>52.3</td><td>28.7</td></tr><tr><td>FreeNoise</td><td>86.2</td><td>54.8</td><td>28.7</td></tr><tr><td>StreamingT2V</td><td>89.2</td><td>46.1</td><td>27.2</td></tr><tr><td>FIFO-Diffusion</td><td>93.1</td><td>57.9</td><td>29.9</td></tr><tr><td>Pyramid Flow</td><td>89.0</td><td>48.3</td><td>24.4</td></tr><tr><td>CausVid (Ours)</td><td>94.9</td><td>63.4</td><td>28.9</td></tr></table>

Table 3. Latency and throughput comparison across different methods for generating 10-second, 120-frame videos at a resolution of $6 4 0 \times 3 5 2$ . The total time includes processing by the text encoder, diffusion model, and VAE decoder. Lower latency (↓) and higher throughput $( \uparrow )$ are preferred.   

<table><tr><td>Method</td><td>Latency (s)</td><td>Throughput (FPS)</td></tr><tr><td>CogVideoX-5B</td><td>208.6</td><td>0.6</td></tr><tr><td>Pyramid Flow</td><td>6.7</td><td>2.5</td></tr><tr><td>Bidirectional Teacher</td><td>219.2</td><td>0.6</td></tr><tr><td>CausVid (Ours)</td><td>1.3</td><td>9.4</td></tr></table>

# 5.2. Ablation Studies

First, we present results of directly fine-tuning the bidirectional DiT into a causal model, without using few-step distillation. We apply causal attention masks to the model and fine-tune it with the autoregressive training method described in Sec. 4.2. As shown in Tab. 4, the many-step causal model performs substantially worse than the original bidirectional model. We observe that the causal baseline suffers from error accumulation, leading to rapid degradation in generation quality over time (orange in Fig. 8 ).

![](images/22.jpg)  
Figure 8. Imaging quality scores of generated videos over 30 seconds. Our distilled model and FIFO-Diffusion are the most effective at maintaining imaging quality over time. The sudden increase of score for the causal teacher around 20s is due to a switch of the sliding window, resulting in a temporary improvement in quality.

We then conduct an ablation study on our distillation framework, examining the student initialization scheme and the choice of teacher model. Tab. 4 shows that given the same ODE initialization scheme (as introduced in Sec. 4.3), the bidirectional teacher model outperforms the causal teacher model and is also much better than the initial ODEfitted model (where we denote the teacher as None). As shown in Fig. 8, the causal diffusion teacher suffers from significant error accumulation (orange), which is then transferred to the student model (green). In contrast, we find that our causal student model trained with our asymmetric DMD loss and a bidirectional teacher (blue) performs much better than the many-step causal diffusion model, highlighting the importance of distillation for achieving both fast and high-quality video generation. With the same bidirectional teacher, we demonstrate that initializing the student model by fitting the ODE pairs can further enhance performance. While our student model improves upon the bidirectional teacher for frame-by-frame quality, it performs worse in temporal flickering and output diversity.

# 5.3. Applications

In addition to text-to-video generation, our method supports a broad range of other applications. We present quantitative results below, with qualitative samples in Fig. 2. We provide additional video results in the supplementary material. Streaming Video-to-Video Translation. We evaluate our method on the task of streaming video-to-video translation, which aims to edit a streaming video input that can have unlimited frames. Inspired by SDEdit [58], we inject noise corresponding to timestep $t _ { 1 }$ into each input video chunk and then denoise it in one step conditioned on the text. We compare our method with StreamV2V [41], a state-of-theart method for this task that builds upon image diffusion models. From 67 video-prompt pairs used in StreamV2V's user study (originally from the DAVIS [62] dataset), we select all 60 videos that contain at least 16 frames. For a fair comparison, we do not apply any concept-specific fine-tuning to either method. Tab. 5 shows that our method outperforms StreamV2V, demonstrating improved temporal consistency due to the video prior in our model.

Table 4. Ablation studies. All models generate videos of 10s. The top half presents results of fine-tuning the bidirectional DiT into causal models without few-step distillation. The bottom half compares different design choices in our distillation framework. The last row is our final configuration.   

<table><tr><td colspan="2">Many-step models</td><td>Causal Generator?</td><td>Pass</td><td># Fwd Temporal Quality</td><td>Frame Quality</td><td>Text Alignment</td></tr><tr><td colspan="2">Bidirectional</td><td>×</td><td>100</td><td>94.6</td><td>62.7</td><td>29.6</td></tr><tr><td colspan="2">Causal</td><td></td><td>100</td><td>92.4</td><td>60.1</td><td>28.5</td></tr><tr><td colspan="2">Few-step models ODE Init. Teacher</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>×</td><td>Bidirectional</td><td></td><td>4</td><td>93.4</td><td>60.6</td><td>29.4</td></tr><tr><td>✓</td><td>None</td><td>:</td><td>4</td><td>92.9</td><td>48.1</td><td>25.3</td></tr><tr><td>✓</td><td>Causal</td><td>✓</td><td>4</td><td>91.9</td><td>61.7</td><td>28.2</td></tr><tr><td>✓</td><td>Bidirectional</td><td>✓</td><td>4</td><td>94.7</td><td>64.4</td><td>30.1</td></tr></table>

Table 5. Evaluation of streaming video-to-video translation.   

<table><tr><td>Method</td><td>Temporal Quality</td><td>Frame Quality</td><td>Text Alignment</td></tr><tr><td>StreamV2V</td><td>92.5</td><td>59.3</td><td>26.9</td></tr><tr><td>CausVid (Ours)</td><td>93.2</td><td>61.7</td><td>27.7</td></tr></table>

Image to Video Generation Our model can perform textconditioned image-to-video generation without any additional training. Given a text prompt and an initial image, we duplicate the image to create the first segment of frames. The model then autoregressively generates subsequent frames to extend the video. We achieve compelling results despite the simplicity of this approach. We evaluate against CogVideoX [96] and Pyramid Flow [28] on the VBench-I2V benchmark, as these are the primary baselines capable of generating 6-10 second videos. As shown in Table 6, our method outperforms existing approaches with notable improvements in dynamic quality. We believe instruction fine-tuning with a small set of image-to-video data could further enhance our model's performance.

# 5.4. Ultra-Long Video Generation

Our model demonstrates strong performance on videos exceeding 10 minutes in duration. As shown in Fig. 9, a 14- minute example video exhibits slight overexposure but retains overall high quality.

Table 6. Evaluation of image-to-video generation. CogVideoX generates 6s video while the other methods generate 10s video.   

<table><tr><td>Method</td><td>Temporal Quality</td><td>Frame Quality</td><td>Text Alignment</td></tr><tr><td>CogVideoX-5B</td><td>87.0</td><td>64.9</td><td>28.9</td></tr><tr><td>Pyramid Flow</td><td>88.4</td><td>60.3</td><td>27.6</td></tr><tr><td>CausVid (Ours)</td><td>92.0</td><td>65.0</td><td>28.9</td></tr></table>

# 6. Discussion

Although our method is able to generate high-quality videos up to 30 seconds, we still observe quality degradation when generating videos that are extremely long. Developing more effective strategies to address error accumulation remains future work. Moreover, while the latency is significantly lower—by multiple orders of magnitude—compared to previous approaches, it remains constrained by the current VAE design, which necessitates the generation of five latent frames before producing any output pixels. Adopting a more efficient frame-wise VAE could reduce latency by an additional order of magnitude, significantly improving the model's responsiveness. Finally, while our method produces high-quality samples using the DMD objective, it comes with reduced output diversity. This limitation is characteristic of reverse KL-based distribution matching approaches. Future work could explore alternative objectives such as EM-Distillation [92] and Score Implicit Matching [55], which may better preserve the diversity of outputs. While our current implementation is limited to generating videos at around 10 FPS, standard engineering optimizations (including model compilation, quantization, and parallelization) could potentially enable real-time performance. We believe our work marks a significant advancement in video generation and opens up new possibilities for applications in robotic learning [13, 89], game rendering [7, 79], streaming video editing [9], and other scenarios that require real-time and long-horizon video generation. Acknowledgment The research was partially supported by the Amazon Science Hub, GIST, Adobe, Google, Quanta Computer, as well as the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions expressed in this document are those of the authors and do not necessarily represent the official policies or endorsements of the United States Air Force or the U.S. Government. Notwithstanding any copyright statement herein, the U.S. Government is authorized to reproduce and distribute reprints for official purposes.

![](images/23.jpg)

# References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. In NeurIPS, 2024. 3 [2] Fan Bao, Chongxuan Li, Yue Cao, and Jun Zhu. All are worth words: a vit backbone for score-based diffusion models. In NeurIPS 2022 Workshop on Score-Based Methods,   
2022. 1 [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1 [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 6 [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.   
2024. 1, 3, 6, 8 [6] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 2,   
3,8 [7] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. arXiv preprint arXiv:2411.00769, 2024. 2, 11 [8] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. arXiv preprint arXiv:2407.01392, 2024. 3, 7, 10 [9] Feng Chen, Zhen Yang, Bohan Zhuang, and Qi Wu. Streaming video diffusion: Online video editing with diffusion models. arXiv preprint arXiv:2405.19726, 2024. 11   
10] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 6   
[11] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In ICLR, 2023.3   
[12] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In NeurIPS, 2022. 8   
[13] Alejandro Escontrela, Ademi Adeniji, Wilson Yan, Ajay Jain, Xue Bin Peng, Ken Goldberg, Youngwoon Lee, Danijar Hafner, and Pieter Abbeel. Video prediction models as rewards for reinforcement learning. NeurIPS, 2024. 11   
[14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 3   
[15] Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, and Alain Rakotomamonjy. Unifying gans and score-based diffusion as generative particle models. In NeurIPS, 2023. 3   
[16] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, and Jun Xiao. Vid-gpt: Introducing gpt-style autoregressive generation in video diffusion models. arXiv preprint arXiv:2406.10981, 2024. 3   
[17] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In ECCV, 2022. 3   
[18] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 2020. 3   
[19] Zekun Hao, Xun Huang, and Serge Belongie. Controllable video generation with sparse trajectories. In CVPR, 2018. 3   
[20] Horace He, Driss Guessous, Yanbo Liang, and Joy Dong. Flexattention: The flexibility of pytorch with the performance of flashattention. PyTorch Blog, 2024. 9   
[21] Jonathan Heek, Emiel Hoogeboom, and Tim Salimans. Multistep consistency models. arXiv preprint arXiv:2403.06807, 2024. 3   
[∠∠] Roberto Henscnel, Levon Knacnatryan, DaniI Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773, 2024. 3, 10   
[23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3, 6   
[24] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022. 1, 6   
[25] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-tovideo generation via transformers. In ICLR, 2023. 6   
[26] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yumig Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 9, 16   
[27] Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. In ICML, 2023. 6   
[28] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024. 1, 2, 3, 9, 10, 11   
[29] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling diffusion models into conditional gans. In ECCV, 2024. 3   
[30] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 6   
[31] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In CVPR, 2024.6   
[32] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. In ICLR, 2024. 3   
[33] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. arXiv preprint arXiv:2405.11473, 2024. 3, 10   
[34] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In NeurIPS, 2021. 6   
[35] Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, and Kurt Keutzer. Streamdiffusion: A pipeline-level solution for real-time interactive generation. arXiv preprint arXiv:2312.12491, 2023. 3   
[36] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. 2024. 2, 3   
[37] Alex X Lee, Richard Zhang, Frederik Ebert, Pieter Abbeel, Chelsea Finn, and Sergey Levine. Stochastic adversarial video prediction. arXiv preprint arXiv:1804.01523, 2018. 3   
[38] Sangyun Lee, Gayoung Lee, Hyunsu Kim, Junho Kim, and Youngjung Uh. Diffusion models with grouped latents for interpretable latent space. In ICML Workshop, 2023. 7   
[39] Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen, and William Yang Wang. T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback. In NeurIPS, 2024. 5   
[40] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In NeurIPS, 2024. 7   
[41] Feng Liang, Akio Kodaira, Chenfeng Xu, Masayoshi Tomizuka, Kurt Keutzer, and Diana Marculescu. Looking backward: Streaming video-to-video translation with feature banks. arXiv preprint arXiv:2405.15757, 2024. 3, 11   
[42] Jian Liang, Chenfei Wu, Xiaowei Hu, Zhe Gan, Jianfeng Wang, Lijuan Wang, Zicheng Liu, Yuejian Fang, and Nan Duan. Nuwa-infinity: Autoregressive over autoregressive generation for infinite visual synthesis. In NeurIPS, 2022. 3   
[43] Shanchuan Lin and Xiao Yang. Animatediff-lightning: Cross-model diffusion distillation. arXiv preprint arXiv:2403.12706, 2024. 5   
[44] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv, 2024. 3   
[45] Haozhe Liu, Wentian Zhang, Jinheng Xie, Francesco Faccio, Mengmeng Xu, Tao Xiang, Mike Zheng Shou, JuanManuel Perez-Rua, and Jürgen Schmidhuber. Faster diffusion via temporal attention decomposition. arXiv e-prints, pages arXiv2404, 2024. 6   
[46] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 3   
[47] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In ICLR, 2024. 3   
[48] Ziwei Liu, Raymond A Yeh, Xiaoou Tang, Yiming Liu, and Aseem Agarwala. Video frame synthesis using deep voxel flow. In CVPR, 2017. 3   
[49] Zhijun Liu, Shuai Wang, Sho Inoue, Qibing Bai, and Haizhou Li. Autoregressive diffusion transformer for textto-speech synthesis. arXiv preprint arXiv:2406.05551, 2024. 7   
[50] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 9   
[51] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022. 6   
[52] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021. 3 [53] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 3 [54] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. In NeurIPS, 2023. 3 [55] Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. arXiv preprint arXiv:2410.16794, 2024. 11 [56] Xiaofeng Mao, Zhengkai Jiang, Fu-Yun Wang, Wenbing Zhu, Jiangning Zhang, Hao Chen, Mingmin Chi, and Yabiao Wang. Osv: One step is enough for high-quality image to video generation. arXiv preprint arXiv:2409.11367,   
2024.5 [57] Michael Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error. In ICLR, 2016. 3 [58] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2021. 11 [59] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, 2023.   
3 [60] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 1, 6, 7, 8 [61] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. 1, 3, 8, 9 [62] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The   
2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 11 [63] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. In ICLR, 2024. 3, 10 [64] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019. 3 [65] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. 2024. 3 [66] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 6 [67] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 6   
[68] David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. Rolling diffusion models. In ICML, 2024. 3   
[69] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 3, 6   
[70] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In ECCV, 2024. 3   
[71] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022. 9   
[72] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 6   
[73] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3, 6, 8   
[74] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In ICLR, 2024. 3   
[75] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In ICLR, 2021. 6   
[76] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 3   
[77] Zhenxiong Tan, Xingyi Yang, Songhua Liu, and Xinchao Wang. Video-infinity: Distributed long video generation. arXiv preprint arXiv:2406.16260, 2024. 3   
[78] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In CVPR, pages 15261535, 2018. 3   
[79] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024. 3, 11   
[80] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 6   
[81] Carl Vondrick and Antonio Torralba. Generating the future with adversarial transformers. In CVPR, 2017. 3   
[82] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-1-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023. 3, 10   
[83] Fu-Yun Wang, Zhaoyang Huang, Qiang Ma, Xudong Lu, Bianwei Kang, Yijin Li, Yu Liu, and Hongsheng Li. Zola: Zero-shot creative long animation generation with short video model. In ECCV, 2024. 3   
[84] Fu-Yun Wang, Zhaoyang Huang, Xiaoyu Shi, Weikang Bian, Guanglu Song, Yu Liu, and Hongsheng Li. Animatelcm: Accelerating the animation of personalized diffusion models and adapters with decoupled consistency learning. arXiv preprint arXiv:2402.00769, 2024. 5   
[85] Xiang Wang, Shiwei Zhang, Han Zhang, Yu Liu, Yingya Zhang, Changxin Gao, and Nong Sang. Videolcm: Video latent consistency model. arXiv preprint arXiv:2312.09109, 2023. 5   
[86] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 3   
[87] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: Highfidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS, 2023. 3   
[88] Wenming Weng, Ruoyu Feng, Yanhui Wang, Qi Dai, Chunyu Wang, Dacheng Yin, Zhiyuan Zhao, Kai Qiu, Jianmin Bao, Yuhui Yuan, et al. Art-v: Auto-regressive text-tovideo generation with diffusion models. In CVPRW, 2024. 3   
[89] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kog Unhi l iiv - for visual robot manipulation. In ICLR, 2024. 11   
[90] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 3   
[91] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models. arXiv preprint arXiv:2410.08151, 2024. 3   
[92] Sirui Xie, Zhisheng Xiao, Diederik P Kingma, Tingbo Hou, Ying Nian Wu, Kevin Patrick Murphy, Tim Salimans, Ben Poole, and Ruiqi Gao. Em distillation for one-step diffusion models. arXiv preprint arXiv:2405.16852, 2024. 11   
[93] Zhening Xing, Gereon Fox, Yanhong Zeng, Xingang Pan, Mohamed Elgharib, Christian Theobalt, and Kai Chen. Live2diff: Live stream translation via uni-directional attention in video diffusion models. arXiv preprint arXiv:2407.08701, 2024. 3   
[94] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In CVPR, 2024. 3   
[95] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 3   
[96] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-tovideo diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 3, 6, 8, 9, 10, 11   
[97] Mingxuan Yi, Zhanxing Zhu, and Song Liu. Monoflow: Rethinking divergence gans via the perspective of wasserstein gradient flows. In ICML, 2023. 3   
[98] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. In ACL, [99] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024. 3, 6, 9   
[100] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024. 3, 6   
[101] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In ICLR, 2023. 6   
[102] Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, and Ming-Yu Liu. Diffcollage: Parallel generation of large content with diffusion models. In CVPR. IEEE, 2023. 3   
[103] Wentian Zhang, Haozhe Liu, Jinheng Xie, Francesco Faccio, Mike Zheng Shou, and Jürgen Schmidhuber. Crossattention makes inference cumbersome in text-to-image diffusion models. arXiv preprint arXiv:2404.02747, 2024. 6   
[104] Zhicheng Zhang, Junyao Hu, Wentao Cheng, Danda Paudel, and Jufeng Yang. Extdm: Distribution extrapolation diffusion model for video prediction. In CVPR, 2024. 3   
[105] Zhixing Zhang, Yanyu Li, Yushu Wu, Yanwu Xu, Anil Kag, Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, Junli Cao, Dimitris Metaxas, et al. Sf-v: Single forward video generation model. arXiv preprint arXiv:2406.04324, 2024. 5   
[106] Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris Metaxas, and Licheng Yu. Avid: Any-length video inpainting with diffusion model. In CVPR, 2024. 3   
[107] Canyu Zhao, Mingyu Liu, Wen Wang, Jianlong Yuan, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655, 2024. 3   
[108] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024. 6   
[109] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 1, 3, 6, 9   
[110] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 7   
[111] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 6   
[112] Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. arXiv preprint arXiv:2410 05317 2024 6

# From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

Supplementary Material

# A. VBench-Long Leaderboard Results

We evaluate CausVid on the VBench-Long dataset using all 946 prompts across 16 standardized metrics. We refer readers to the VBench paper [26] for a detailed description of the metrics. As shown in Tab. 7, our method achieves state-of-the-art performance with the highest total score of 84.27. The radar plot in Fig. 10 visualizes our method's comprehensive performance advantages. Our method is significantly ahead in several key metrics including dynamic degree, aesthetic quality, imaging quality, object class, multiple objects, and human action. More details can be found on the official benchmark website (ht tps : / / huggingface.co/spaces/Vchitect/vBench_ Leaderboard).

![](images/24.jpg)  
dimensions.

<table><tr><td>Method</td><td></td><td>Quality Score</td><td></td><td>Subject Con</td><td></td><td>98.57</td><td></td><td>98.98</td><td>Dynamic Degree 63.89</td><td>thetic Quality Imaging Quality 65.35</td><td>Object Class 86.61</td><td>Multiple Objects 68.84</td><td></td><td>Human Action 87.04</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vchitect Jimeng</td><td>82.24</td><td>83.29 83.54</td><td>77.06 76.69</td><td>96.83 97.25</td><td>96.66 98.39</td><td>99.03</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>97.20</td><td></td><td></td><td>56.57</td><td>23.73</td><td>24.7 25.01</td><td>27.10 27.57</td></tr><tr><td>CogVideoX Vidu</td><td>81.67</td><td>82.75</td><td>77.04</td><td>96.23</td><td>96.52</td><td></td><td></td><td>98.09 9.71 96.92</td><td>38.43 70.97</td><td>68.80 61.98 60.87</td><td>67.09 6.22 62.90</td><td>89.62 85.23 88.43</td><td>69.08 61.6055 62.11</td><td>90.10</td><td>89.05 82.81</td><td>77.45 66.35 66.18</td><td>44.94 53.20 46.07</td><td>22.27 24.91 21.54</td><td>25.38</td><td>27.59</td></tr><tr><td>Kling</td><td>8.99</td><td>83.85 83.39</td><td>74.04 75.68</td><td>94.63 98.33</td><td>96.55 97.60</td><td></td><td></td><td></td><td></td><td>61.21</td><td></td><td>87.24</td><td></td><td></td><td>87.24 </td><td>73.03</td><td>50.86</td><td>19.62</td><td>23.79 24.17</td><td>26.47 26.42</td></tr><tr><td>CogVideoX1.5-5B Gen-3</td><td>82.17 82.32</td><td>82.78 84.11</td><td>79.76 75.17</td><td>96.87 97.10</td><td>97.35 96.62</td><td></td><td></td><td>98.31 99.23</td><td>N6G 60.14</td><td>62.79 63.34</td><td></td><td>87.47</td><td>69.65 53.64</td><td>96.40</td><td>87.55 80.90</td><td>80.25 65.09</td><td>5.91 </td><td>24.89</td><td>25.19</td><td>27.30 26.69</td></tr><tr><td>CausVid (Ours)</td><td>84.27</td><td>85.65</td><td>78.75</td><td>97.53</td><td>97.19</td><td></td><td>96.24</td><td>98.055</td><td>92.69</td><td>64.15</td><td>G. </td><td>$7.81 </td><td>72.15</td><td>99.80</td><td>80.17</td><td>64.65</td><td>56.58</td><td>24.27 24.31</td><td>25.33 24.71</td><td>27.51</td></tr></table>

TaFu  e-L eT le . Which video is more aesthetic and faithfully follow the prompt shown above? (26 models remaining). Label with beautiful photography, depth of field.

![](images/25.jpg)  
left/right arrangement.