# MAGI-1: Autoregressive Video Generation at Scale

Sand AI research@sand.ai

# Abstract

We present MAGI-1, a world model that generates videos by autoregressively predicting a sequence of video chunks, defined as fixed-length segments of consecutive frames. Trained to denoise per-chunk noise that increases monotonically over time, MAGI-1 enables causal temporal modeling and naturally supports streaming generation. It achieves strong performance on image-to-video (I2V) tasks conditioned on text instructions, providing high temporal consistency and scalability, which are made possible by several algorithmic innovations and a dedicated infrastructure stack. MAGI-1 facilitates controllable generation via chunk-wise prompting and supports real-time, memory-efficient deployment by maintaining constant peak inference cost, regardless of video length. The largest variant of MAGI-1 comprises 24 billion parameters and supports context lengths of up to 4 million tokens, demonstrating the scalability and robustness of our approach. The code and models are available at magi-source and magi-attention. The product can be accessed at magi -product.

# 1 Introduction

World modeling and video generation have emerged as central challenges in artificial intelligence, requiring the synthesis of temporally coherent and photorealistic sequences conditioned on semantically rich inputs such as natural language, static imagery, or short vcis. This task reside t the ntersion  spatal pereption nd temporal , with profound implications for fields including robotics, embodied artificial intelligence, interactive media, and scientific simulation. As video becomes a dominant modality for both human communication and machine understanding, the demand for generative models that are not only high-fidelity and computationally efficient, but also causally consistent and compatible with streaming applications, has become increasingly urgent.

Building on the remarkable success of diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020) and flow-matching frameworks (Lipman et al., 2022; Liu et al., 2022a) in image generation, recent research has increasingly focused on extending these approaches to video synthesis. However, most large-scale video diffusion models continue to rely on globally conditioned denoising architectures that process the entire temporal sequence simultaneously. These models typically employ uniform noise levels and require fullsequence access during inference." Such designs disregard the causal structure inherent to temporal data, rendering them suboptimal for scenarios requiring streaming, real-time interaction, or autoregressive generation.

To overcome these limitations, we present MAGI-1: a large-scale diffusion-based generative model that produces video through the autoregressive generation of temporally segmented chunks, each consisting of a fixed-length sequence of consecutive frames. This chunk-wise approach offers a principled trade-off between causal modeling and temporal abstraction, enabling the model to capture mid-range temporal dependencies while maintaining strict left-to-right temporal consistency. Training is conducted at the chunk level with temporally progressive noise levels, resulting in a model that is both autoregressively structured and adaptable in its conditional generation capacity.

MAGI-1 adheres strictly to causal constraints and facilitates real-time, streaming-compatible video synthesis that approximates multi-step diffusion trajectories with reduced-step, chunklevel predictions. This is enabled by a Transformer (Vaswani et al., 2017) backbone specifically designed for bidirectional spatial and causal temporal denoising, supported by a carefully engineered training infrastructure. Central to this infrastructure is a novel distributed attention mechanism (MagiAttention) tailored for ultra-long autoregressive contexts, along with a scalable execution framework optimized for low-latency, parallelized inference. These core components are further augmented by a robust data curation pipeline that supports multi-stage training and dynamically adapts the data distribution based on ongoing model evaluation. Together, these architectural and algorithmic advances empower MAGI-1 to deliver efficient, scalable, and controllable video generation. Notably, the inference-time peak resource usage of MAGI-1 is independent of the total video length, as each chunk is processed with a fixed computational and memory footprint. This makes MAGI-1 particularly suitable for low-latency, memory-efficient applications. The largest variant of the model comprises 24 billion parameters and supports context lengths of up to 4 million tokens, demonstrating the scalability and robustness of the framework.

We evaluate MAGI-1 using both internal metrics and publicly available benchmarks, with a particular focus on the image-to-video (I2V) generation task. Our evaluation protocol assesses prompt fidelity, temporal coherence, and subject integrity. On VBench-I2V (Huang et al., 2024) and Physics-IQ Benchmark (Motamed et al., 2025), MAGI-1 achieves substantial improvements over previous models, especially in its ability to synthesize complex motion, preserve semantic alignment, and model physically plausible interactions.

In summary, MAGI-1 establishes a scalable and autoregressive foundation for diffusionbased video synthesis. By integrating architectural innovations, high-throughput inference techniques, and a comprehensive data processing framework, MAGI-1 bridges the gap between high-quality generative performance and real-time applicability. The complete inference codebase and pre-trained models are publicly accessible at magi-source, the distributed attention available at magi-attention, and a live demonstration available at magi-product.

# 2 MAGI-1

![](images/1.jpg)  
Figure 1: (Left) MAGI-1 performs chunk-wise autoregressive denoising. The video is generated in chunks of 24 frames, where each chunk attends to all previously denoised chunks. Once a chunk reaches a certain denoising level, the next chunk begins generation. (Right) A block-causal attention mask enforces temporal causality across chunks, enabling pipelined and parallel generation.

MAGI-1 is an autoregressive denoising video generation model operating in latent space. The generation process is illustrated in Fig. 1. Unlike other bi-directional denoising models (e.g., Sora (OpenAI, 2024)) that generates the video as a whole, MAGI-1 generates the video chunk-by-chunk in a pipeline manner. Specifically, each chunk consists of multiple frames that are denoised holistically. As a chunk is denoised to a certain extent (not necessary completely clean), the next chunk begins generation, conditioned to all preceding chunks. This design allows multiple chunks to be processed concurrently. In our implementation, each chunk contains 24 raw frames (equivalent to one second video clip at 24 FPS), and up to four chunks can be inferred simultaneously.

Compared to fully denoising one chunk before starting subsequent chunks, our method leverages parallelism to better utilize computation, reducing the latency of obtaining subsequent clean chunks and enabling real-time streaming video generation. Moreover, the auto-regressive design naturally supports video continuation without additional specific designs, and extends seamlessly to image-to-video generation. This unified framework enables us to cover text-to-video generation, video continuation, and image-to-video generation within a single pre-training process, eliminating the need for task-specifc fine-tuning required by other methods. By maintaining consistency between pre-training and downstream tasks, our approach achieves superior performance in both video continuation and image-to-video generation.

In this section, we will systematically introduce the training, distillation, and inference of MAGI-1 in detail.

# 2.1 Transformer-based Variational Auto-Encoder

To improve the efficiency of both training and inference, MAGI-1 employs a variational autoencoder (VAE) to obtain a compressed latent space, over which denoising is performed. While most open-source VAEs are built upon convolutional architectures (e.g., U-Net (Ronneberger et al., 2015)), they are considerably slower than the transformer-based counterparts (e.g., ViT (Dosovitskiy et al., 2020)) of comparable model size on modern GPUs. To address this, we design our VAE architecture based on transformers.

The architecture of our VAE is illustrated in Fig. 2. In the encoder, the input is first processed by an embedding module based on a 3D convolution with a kernel size of $\bar { 8 } \times 8 \times 4 ^ { 1 }$ and a stride of $8 \times 8 \times 4 ,$ producing an output with 1024 channels. Absolute positional embeddings are then added to enrich spatial and temporal representations. Building on this, we stack 24 transformer blocks, where self-attention is stabilized through query, key and value normalization to improve training stability. The output of the final transformer block is normalized by a LayerNorm and then projected via a linear layer to 32 channels: the first 16 channels represent the predicted mean, and the remaining 16 channels represent the predicted log-variance. Compared to the raw video input, the encoded features are downsampled by a factor of 8 in the spatial dimensions and by a factor of 4 in the temporal dimension.

The decoder adopts a symmetric architecture to the encoder. To restore the original spatial and temporal resolution, we first apply a pixel shuffle operation to the output of the final transformer block, followed by a 3D convolution with a kernel size of $3 \times 3 \times \bar { 3 }$ and 3 output channels to generate the final output in pixel space. For image inputs consisting of a single frame, we replicate the frame four times along the temporal dimension, which yields better performance compared to padding with three empty frames.

<table><tr><td rowspan=1 colspan=1>VAE</td><td rowspan=1 colspan=1>PSNR</td><td rowspan=1 colspan=1>Params(M)</td><td rowspan=1 colspan=1>Avg Encode Time(ms)</td><td rowspan=1 colspan=1>Avg Decode Time(ms)</td></tr><tr><td rowspan=3 colspan=1>OpenSoraPlan-1.2 (Lin et al., 2024)CogVideoX (Yang et al., 2025)HunyuanVideo (Kong et al., 2024)StepVideo (Ma et al., 2025)Wan2.1 (Wang et al., 2025a)Ours</td><td rowspan=3 colspan=1>28.3935.9937.2733.7535.9536.55</td><td rowspan=1 colspan=1>239</td><td rowspan=1 colspan=1>51.08</td><td rowspan=3 colspan=1>17.48142.9647.1118.1279.4312.28</td></tr><tr><td rowspan=1 colspan=1>216</td><td rowspan=2 colspan=1>40.19124.3930.4751.9136.68</td></tr><tr><td rowspan=1 colspan=1>246499127614</td></tr></table>

Table 1: Comprehensive comparison of our VAE with other open-source approaches. Thanks to the optimized inference support of transformers, our VÅE achieves the fastest decode speed under identical hardware conditions, despite having the largest model size.

The training process of the VAE consists of two stages. In the first stage, we use a fixed input resolution during training: 16-frame short clips with a spatial resolution of $2 5 6 \times 2 5 6$ pixels, to maximize training efficiency by avoiding unnecessary padding. In the second stage, two key modifications are introduced. First, both image data (single frame) and video data (16-frame clip) are jointly used during training. Second, we adopt variable spatial resolutions and aspect ratios by randomly sampling at each training step, enabling the VAE to generalize across different resolutions. Specifically, we constrain the total number of pixels (height $\times$ width) is approximately $2 5 6 ^ { \dot { 2 } }$ or $3 8 4 ^ { 2 }$ ,while sampling the aspect ratio uniformly from the range [0.25, 4.0]. In both stages, we apply a combination of L1 loss, KL divergence loss, LPIPS loss, and GAN loss, following common practice.

During inference, we use sliding window approach to support arbitrary resolutions. In the spatial dimension, we adopt a window size of $2 5 6 \times 2 5 6$ pixels with a stride of 192 pixels, resulting in a $2 5 \%$ overlap between adjacent patches in spatial. In the temporal dimension, no overlap is applied.

Tab. 1 shows the comparison with other open-source VAEs. All models were evaluated on a single NVIDIA H800 GPU. To eliminate potential biases from varying slicing strategies at higher resolutions, we report the average processing speed measured across 169 test videos, each containing 25 frames with a spatial resolution of $2 5 6 \times 2 5 6$ pixels. Despite having the larget model size, our transormer-based VAE achives the astest average decoding time among all models and significantly outperforms most baselines in encoding speed. In terms of reconstruction quality (measured by PSNR), it remains highly competitive, ranking second overall.

![](images/2.jpg)  
Figure 2: Model Architecture of Transformer-based VAE.

# 2.2 Auto-Regressive Denoising Model

# 2.2.1 Training objective

MAGI-1 employes flow-matching (Albergo & Vanden-Eijnden, 2022; Liu et al., 2022a; Lipman et al., 2022) as its training objective. Given a training video clip contains $n$ chunks, we sample independent Gaussian noises for each chunk. The linear interpolation with respect to the denoising timestep $t$ between the sampled noise and the clean latent of the $i \cdot$ -th chunk is defined as:

$$
x _ { i } ^ { t } = ( 1 - t ) x _ { i } ^ { 0 } + t x _ { i } ^ { 1 } ,
$$

where $x _ { i } ^ { 1 }$ denotes the latent of $i .$ th chunk and $x _ { i } ^ { 0 }$ is the corresponding sampled Gaussian noise. The ground-truth velocity for each chunk is given by:

$$
v ^ { * } ( x _ { i } ^ { t } ) = \frac { d x _ { i } ^ { t } } { d t } = x _ { i } ^ { 1 } - x _ { i } ^ { 0 } .
$$

In the auto-regressive model, earlier chunks are cleaner than later ones. For convenience, we define the noise timestep sampled assigned to each chunk as $t _ { i } ,$ and impose the constraint $t _ { i } ~ < ~ t _ { j }$ whenever $i < j$ 2 The interpolation of the entire video clip is then defined as: $X _ { T } = \big \{ x _ { 0 } ^ { t _ { 0 } } , x _ { 1 } ^ { t _ { 1 } } , . . . , x _ { n } ^ { t _ { n } } \big \}$ T

$$
\mathbb { E } _ { c , X _ { T } } \parallel { \boldsymbol v } ( x _ { i } ^ { t _ { i } } | t _ { i } , c , \{ x _ { j < i } ^ { t _ { j } } \} ; \theta ) - { \boldsymbol v } ^ { * } ( x _ { i } ^ { t _ { i } } ) \parallel ^ { 2 } .
$$

where $v ( \cdot ; \theta )$ is the denoising model parameterized by $\theta _ { . }$ , and $c$ denotes the conditioning text inputs. Note that the prediction of velocity for $x _ { i }$ explicitly conditioned on all its preceding chunks $x _ { j }$ where $j < i$ .

In contrast, typical bi-directional denoising video models do not enforce monotonicity of the noise timestep. Instead, they apply the equality constraint, where all chunks share the same noise timestep. Accordingly, their training objective is formulated as:

$$
\mathbb { E } _ { c , X _ { T } } \parallel v ( x _ { i } ^ { t _ { i } } | t _ { i } , c , X _ { T } ; \theta ) - v ^ { * } ( x _ { i } ^ { t _ { i } } ) \parallel ^ { 2 } .
$$

where the velocity prediction for $x _ { i }$ is conditioned on all chunks, regardless their temporal order.

![](images/3.jpg)  
Figure 3: Model Architecture of Auto-Regressive Denoising Model.

# 2.2.2 Model Architecture

MAGI-1 is built upon the Diffusion Transformer (DiT) architecture. However, to better meet the requirements of auto-regressive modeling and to improve training efficiency and stability at scale, we introduce several key modifications. As shown in Fig. 3(a), MAGI1 follows a high-level architecture similar to the standard DiT, consisting of four main components: patch embedding, attention, feed-forward network (FFN), and final stem. We employ T5 (Raffel et al., 2020) to extract text embeddings, while the timestep information is encoded using sinusoidal positional embeddings. Our primary modifications target the attention and FFN modules, which are illustrated in Fig. 3(b) and Fig. 3(c), respectively. In the following, we provide a detailed description of these modifications.

Block-Causal Attention MAGI-1 employs full attention within each chunk and causal attention across chunks. Spatial and temporal positional information is encoded using a learnable 3D RoPE (Su et al., 2024), in which the base frequency is learnable. However, existing attention implementations (Dao et al., 2022; Dao, 2023) do not efficiently support blockcausal attention, therefore, we implemented a new kernel called Flexible-Flash-Attention on top of FlashAttention-3. Further details can be found in Sec. 4.1.2.

Parallel Attention Block MAGI-1 adopts a parallel design for spatial-temporal selfattention and cross-attention with external conditioning input, offering improved computational efficiency over the serial attention architecture. In the serial setup, each attention module independently computes query projections and incurs a separate round of Tensor Parallel (TP) communication. In contrast, the parallel block computes query projections once and applies them to both attention types concurrently, reducing TP communication from two rounds to one per block. This optimization lowers inter-GPU synchronization overhead and enhances scalability in large-scale models.

QK-Norm and GQA Earlier studies on vision transformers (Liu et al., 2022b; Dehghani et al., 2023) have shown that normalizing the queries and keys of attention can significantly improve training stability. Moreover, inspired by recent advances in large language models (LLMs), we replace the standard multi-head attention (MHA) with grouped-query attention (GQA) (Ainslie et al., 2023) to reduce memory consumption. Both techniques are applied to the spatial-temporal attention and cross-attention modules in our design.

Sandwich Normalization in FFN In practice, we have noticed that the numerical problems are more likely to appears in FFN modules as the model size increase. Therefore, we have added LayerNorm before and after the FFN input and output to alleviate the challenge.

SwiGLU SwiGLU (Shazeer, 2020) has been widely adopted in large language models and has been shown to consistently improve performance than ReLU. Therefore, we employ SwiGLU in the feed-forward network (FFN) of our 24B model.

Softcap Modulation The standard DiT incorporates timestep information via adaLN, where the denoising timestep is used to compute a scaling factor that modulate both the input and output activations of the attention and FFN. While this design works well for small models, we observed that in large models it tends to amplify activation magnitudes factor, constraining its values within the range of $\left[ - 1 , 1 \right]$ Furthermore, since we adopt QK-Norm in attention modules, we remove the input modulation of adaLN.

# 2.2.3 Training Recipes

Training Configurations We train a 4.5B and 24B MAGI-1 models and their configurations is shown in Tab. 2. The training is organized into three stages. Take the 4.5B model as an example. In the first two stages, the resolution of training data is set to $3 6 0 \mathrm { p }$ and $4 8 0 \mathrm { p } ,$ respectively, with video length is up to 8 seconds. In the third stage, the resolution is further increased to $7 2 0 \mathrm { p } .$ , and the video length is extended up to 16 seconds. Throughout all three stages, the image and video are trained jointly. At the beginning of training, we apply a learning rate warmup, gradually increasing the learning rate to 1e—4 in 1000 steps. Then, we adopt a stepwise learning rate scheduling strategy. In the first two stages, the learning rate remains constant, and the stage is switched when the visual assessment of generated video does not significantly improve. In the third stage, we gradually reduce the learning rate once the validation loss reaches a plateau, eventually reducing to 1e—5.

Table 2: Model Specification of MAGI-1.   

<table><tr><td></td><td>4.5B</td><td>24B</td></tr><tr><td>Layers</td><td>34</td><td>48</td></tr><tr><td>Model Dimension</td><td>3072</td><td>6144</td></tr><tr><td>FFN Activation</td><td>GLU</td><td>SwiGLU</td></tr><tr><td>FFN Dimension</td><td>12288</td><td>16384</td></tr><tr><td>Attention Type</td><td>GQA + QK-Norm</td><td>GQA + QK-Norm</td></tr><tr><td>Block-Casual Attention Head</td><td>128</td><td>128</td></tr><tr><td>Block-Casual Attention Group</td><td>8</td><td>8</td></tr><tr><td>Cross Attention Head</td><td>128</td><td>128</td></tr><tr><td>Cross Attention Group</td><td>8</td><td>8</td></tr><tr><td>Positional Embedding</td><td>Learnable 3d RoPE</td><td>Learnable 3d RoPE</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Weight Decay</td><td>1 × 10−1</td><td>1 × 10−1</td></tr><tr><td>Peak LR</td><td>1 × 10−4</td><td>1 × 10−4</td></tr><tr><td>Warm-up</td><td>1000</td><td>10000</td></tr><tr><td>β1</td><td>0.9</td><td>0.9</td></tr><tr><td>β2</td><td>0.95</td><td>0.95</td></tr></table>

For the 24B model, we reduce the resolution in the first training stage from $3 6 0 \mathrm { p }$ t $2 5 6 \mathrm { p } ,$ as this stage primarily serves to making the model learn global motion dynamics and semantic concepts. Lowering the resolution allows for more training iterations within the same computational budget, thereby improving training efficiency. In addition, we extend the learning rate warmup phase to 10,000 steps to enhance stability during the early training phase. Furthermore, since larger models typically require longer training to reach performance saturation, we proportionally increase the number of training steps at each stage, guided by empirical visual assessment on the validation set.

Multi-Task Training via Data Configurations Bi-directional denoising models typically support only text-to-video generation during pretraining, while tasks such as image-tovideo generation often require dedicated architectural designs or additional finetuning. In contrast, within the auto-regressive framework, text-to-video, image-to-video, and video continuation tasks differ solely in the proportion of clean chunks present in the training data. As illustrated in Fig. 4, the early stage of text-to-video generation corresponds to the case of all chunks are noisy, while the inclusion of some clean chunks represents to video continuation. Image-to-video generation is a special case of video continuation, with only the first frame of the first chunk being clean.

Thanks to this property, our auto-regressive model enables unification of various generation tasks under a single training objective without additional task specific fine-tuning and requiring only adjustment of the proportion of clean chunks in the training data.

Furthermore, unlike bi-directional denoising models — where the text condition must be predefined for the entire video and remains fixed throughout generation — MAGI-1 allows for different text conditions to be provided for each chunk, enabling fine-grained, chunkwise text control. To better support this capability, we design a dedicated auto-regressive captioning strategy (Data details are described in Sec. 3.4) that adapts training accordingly. Additional examples of this fine-grained control are provided in Sec. 2.6.

Timestep Sampler in Training Early studies have demonstrated that improving the design of timestep sampler (commonly known as SNR sampler) can facilitate training efficiency by better allocating computations across different noise levels. (Esser et al., 2024) introduce the Logit-Normal sampling strategy, which provides a flexible framework for controlling the distribution of sampled timestep, the transformed timestep density $\pi ( t )$ is:

![](images/4.jpg)  
Figure 4: The figure shows how different tasks can be unified by varying the proportion of clean chunks. Each vertical bar represents a latent frame in a chunk, with darker bars indicating higher noise levels and the white bars denoting clean frames. The first row illustrates the early inference stage of T2V generation, starting from a single fully noisy chunk and progressing to multiple noisy chunks, before any clean chunk has been produced. The middle row depicts the case of I2V generation, treated as a special case of continuation in which only the first frame of the first chunk is clean. The last row describes a general stage where clean chunks are already available, applicable to video continuation and other scenarios involving prior denoised content.

![](images/5.jpg)  
Figure 5: The probability density of training timestep. We generally aim to allocate $7 0 \%$ of training computation when $\mathrm { t } < \bar { 0 . 3 }$ .

$$
\pi ( t ; m , s ) = \frac { 1 } { s \sqrt { 2 \pi } } \frac { 1 } { t ( 1 - t ) } \exp ( - \frac { ( \mathrm { l o g i t } ( t ) - m ) ^ { 2 } } { 2 s ^ { 2 } } ) ,
$$

where $\begin{array} { r } { \log \mathrm { i t } ( t ) = \log \frac { t } { 1 - t } } \end{array}$ In addition, (Esser et al., 2024) further introduce a timestep shift strategy to handle the resolution increasing:

$$
t ^ { \prime } = { \frac { w t } { 1 - ( 1 - w ) t } }
$$

In MAGI-1, we draw inspiration from these two sampling strategies but make adjustment for video data. Since videos typically contain more redundant information than images, we aim to shift the overall sampling distribution further towards the noise side compared to images. In our preliminary experiments, as shown in Fig. 6, we observed that the model is capable of generating reasonably clear video outputs at $\bar { t } = 0 . 3$ .Based on this observation, we heuristically allocate approximately $7 0 \%$ of the training computation budget to the region where $t < 0 . 3$ .Following this principle, we set the $m = 0$ , $s \stackrel { - } { = } 0 . 5$ , and $w \stackrel { } { = } 1 / 3$ for all cases during training.

![](images/6.jpg)  
Figure 6: The generation results at the given timestep t. Through empirical experiments, we found that model is capable of generating quite clear video outputs at $t = 0 . { \dot { 3 } }$ .

Design Choices for Clean Chunks There are two types of chunks in the training of MAGI-1: noisy chunks and clean chunks, and we adopt three key designs to handle clean chunks:

First, in practical video continuation scenarios, users typically upload an initial video clip and provide follow-up text descriptions or dynamically update the prompt during the continuation process. Considering this usage, we argue that clean chunks should not be conditioned on text inputs.

Second, exposure bias is a well-recognized challenge in training auto-regressive models (Bengio et al., 2015; Wiseman & Rush, 2016). A common mitigation strategy is to inject a small amount of noise into clean data during training. Following this practice, we inject up to $5 \%$ noise into clean chunks to alleviate exposure bias.

Finally, since clean chunks are relatively abundant in the pre-training data, they risk dominating the training signal. To address this, we apply the loss function exclusively to noisy chunks. Nevertheless, clean chunks still participate in training through the attention mechanism and continue to receive gradient updates. Empirically, we observe that blocking the gradients of clean chunks leads to a significant degradation in model performance.

# 2.3 Distillation Using Shortcut Model

Flow-matching formulates the generative process as an ODE that maps noise to data along high-dimensional, curved trajectories. Sampling from such models is computationally intensive, typically requiring dozens of function evaluations with sufficiently small step sizes to incrementally transform noise into data. This inefficiency motivates the development of diffusion distillation methods (Luhman & Luhman, 2021; Salimans & Ho, 2022) that can reduce the required number of inference steps without sacrificing sample quality.

This work adopts shortcut model (Frans et al., 2024) as the distillation target. Given a noisedata interpolation defined by $x _ { i } ^ { t } = ( 1 - t ) x _ { i } ^ { 0 } + t x _ { i } ^ { 1 } .$ where $x _ { i } ^ { 1 }$ is the clean data point at the $i$ -th chunk and $x _ { i } ^ { 0 }$ denotes Gaussian noise, the shortcut model uses a single neural network to predict a velocity field $v \big ( x _ { i } ^ { t } \mid t , s \big ) ^ { 3 }$ , conditioned not only on the current timestep $t _ { \iota }$ but also on the desired step size $s \propto \Delta t$ Here, $\Delta t$ denotes the interval between adjacent timesteps, while the reciprocal $1 / s \in 2 ^ { \mathbb { N } }$ specifies the number of function evaluations required to complete the denoising process4.

The generation process of the shortcut model closely resembles the flow-matching formulation and can be expressed as $\hat { x } _ { i } ^ { t + \Delta t } = x _ { i } ^ { t } + \Delta t \cdot v ( x _ { i } ^ { t } , t , s ) .$ where $\hat { x } _ { i } ^ { t + \Delta t }$ denotes the model-predicted next point in the denoising trajectory, explicitly indicated by the hat symbol over $x _ { i } ^ { t + \Delta t }$ . As $\Delta t \to 0$ ,thi ulati reover the tanar flow-matcigsario, where the shortcut model approximates the instantaneous velocity.

During training, the shortcut model constructs distillation targets using a bootstrap procedure, leveraging the principle that a single shortcut step is equivalent to two consecutive $\hat { x } _ { i } ^ { t + \Delta t _ { 1 } + \Delta t _ { 2 } } = x _ { i } ^ { t } + \left( \Delta t _ { 1 } + \Delta t _ { 2 } \right) \cdot v ( x _ { i } ^ { t } , t , 2 s )$ can also be written as +Δ1+Δ $\hat { x } _ { i } ^ { t + \Delta t _ { 1 } + \Delta t _ { 2 } } = \hat { x } _ { i } ^ { t + \Delta t _ { 1 } } + \Delta t _ { 2 } \cdot v ( \hat { x } _ { i } ^ { t + \Delta t _ { 1 } } , t + \Delta t _ { 1 } , s ) = x _ { i } ^ { t } + \Delta t _ { 1 } \cdot v ( x _ { i } ^ { t } , t , s ) +$ $\Delta t _ { 2 } \cdot v ( \hat { x } _ { i } ^ { t + \Delta t _ { 1 } } , t + \Delta t _ { 1 } , s )$ eag o he reltiohip:

$$
v ( x _ { i } ^ { t } , t , 2 s ) = \frac { \Delta t _ { 1 } } { \Delta t _ { 1 } + \Delta t _ { 2 } } v ( x _ { i } ^ { t } , t , s ) + \frac { \Delta t _ { 2 } } { \Delta t _ { 1 } + \Delta t _ { 2 } } v ( \hat { x } _ { i } ^ { t + \Delta t _ { 1 } } , t + \Delta t _ { 1 } , s )
$$

In practice, the smallest $s$ utilized is $1 / 6 4 ,$ corresponding to the standard flow-matching inference setting that requires 64 function evaluations. When training with this minimal step size, we incorporate classifier-free guidance (CFG) distillation (Meng et al., 2023) (see Sec. 2.4.1 for details). The step size $s$ for distillation is cyclically sampled from the set $[ 1 / 6 4 ] \times 8 \cup [ 1 / 3 2 , 1 / 1 6 , 1 / 8 ]$ This sampling strategy enables a single distilled model to perform denoising with different computational budgets (64, 32, 16, or 8 steps), thus providing flexibility to dynamically balance generation quality and inference efficiency at test time.

# 2.4 Inference Approach

# 2.4.1 Diffusion Guidance

Classifier-free guidance (Ho & Salimans, 2022), a widely adopted low-temperature sampling technique in diffusion models, offers a principled approach to mediating the inherent trade-off between sample fidelity and diversity in generative modeling. This technique is particularly effective in text-to-video generation, where the objective is to synthesize temporally coherent video frames that conform to given textual prompts.

To improve clarity, we omit irrelevant variables and express the guided posterior distribution of the latent variable $x _ { t }$ given condition $c$ using Bayes' rule as $p _ { \mathrm { g u i d e d } } \mathbf { \bar { ( } } x _ { t } \mid c ) \propto p ( x _ { t } ) \cdot p ( c \mid$ $x _ { t } ) ^ { w }$ , where the exponent $w \ge 1$ serves as an inverse temperature parameter. Exponentiating the conditional likelihood $p ( c \mid x _ { t } )$ concentrates the distribution around modes better aligned with the conditioning signal, thereby reducing entropy and improving sample fdelity.

In our setting, the generation of the $i \cdot$ th video chunk $x _ { i }$ is conditioned not only on the textual prompt $c _ { \mathrm { t e x t } } ,$ but also on a sequence of preceding chunks $x _ { < i } ,$ which may include partially denoised or fully noised representations. The guided conditional distribution is thus formulated as:

$$
p _ { \mathrm { g u i d e d } } ( x _ { i } \mid x _ { < i } , c _ { \mathrm { t e x t } } ) \propto p ( x _ { i } ) \cdot p ( x _ { < i } \mid x _ { i } ) ^ { w _ { \mathrm { p r e v } } } \cdot p ( c _ { \mathrm { t e x t } } \mid x _ { < i } , x _ { i } ) ^ { w _ { \mathrm { t e x t } } } ,
$$

where $w _ { \mathrm { p r e v } }$ and $w _ { \mathrm { t e x t } }$ are scalar weights modulating the influence of temporal and semantic signals, respectively. Taking the logarithm of both sides of Eq. 8, we obtain the guided score: $\begin{array} { r } { \nabla _ { x _ { i } } \dot { \log { p _ { \mathrm { g u i d e d } } ( x _ { i } \mid x _ { < i } , c _ { \mathrm { t e x t } } ) } ^ { \circ } } = \nabla _ { x _ { i } } \log { p ( x _ { i } ) } + w _ { \mathrm { p r e v } } \cdot \dot { \nabla } _ { x _ { i } } \log { p ( x _ { < i } \mid x _ { i } ) } + \dot { w } _ { \mathrm { t e x t } } . } \end{array}$ . $\nabla _ { x _ { i } } \log p \big ( c _ { \mathrm { t e x t } } \mid x _ { < i } , x _ { i } \big )$ Applying Bayes' rule, we rewrite the gradients as $\nabla _ { x _ { i } } \log { p ( \boldsymbol { x } _ { < i } \ | }$ $x _ { i } ) \ = \ \nabla _ { x _ { i } } \log p ( x _ { i } \mid x _ { < i } ) - \nabla _ { x _ { i } } \log p ( x _ { i } ) ,$ and $\nabla _ { x _ { i } } \log { p ( c _ { \mathrm { t e x t } } \mid x _ { < i } , x _ { i } ) } = \nabla _ { x _ { i } } \log { p ( x _ { i } \mid }$

$$
) { w _ { \mathrm { p r e v } } } = 1 . 5
$$

![](images/7.jpg)  
Figure 7: This figure demonstrates the impact of $w _ { \mathrm { p r e v } }$ on the generation results. (a) When $w _ { \mathrm { p r e v } } = 1 . 0 ,$ there are perceptible misalignments between adjacent chunks (e.g., the shape of the smoke). (b) When $\begin{array} { r } { \dot { w } _ { \mathrm { p r e v } } = 1 . 5 } \end{array}$ , this phenomenon is significantly alleviated.

$x _ { < i } , c _ { \mathrm { t e x t } } ) - \nabla _ { x _ { i } } \log p ( x _ { i } \mid x _ { < i } )$ .Substituting and regrouping, we arrive at the final guided score:

$$
\begin{array} { r l } { \nabla _ { x _ { i } } \log p _ { \mathrm { g u i d e d } } ( x _ { i } \mid x _ { < i } , c _ { \mathrm { t e x t } } ) = } & { ( 1 - w _ { \mathrm { p r e v } } ) \cdot \nabla _ { x _ { i } } \log p ( x _ { i } ) } \\ { + } & { ( w _ { \mathrm { p r e v } } - w _ { \mathrm { t e x t } } ) \cdot \nabla _ { x _ { i } } \log p ( x _ { i } \mid x _ { < i } ) } \\ { + } & { w _ { \mathrm { t e x t } } \cdot \nabla _ { x _ { i } } \log p ( x _ { i } \mid x _ { < i } , c _ { \mathrm { t e x t } } ) . } \end{array}
$$

This decomposition cleanly separates the contributions of the unconditional prior, temporal context, and prompt conditioning. It enables controllable trade-offs between coherence and semantic fidelity in autoregressive generation.

As a special case, when $w _ { \mathrm { p r e v } }$ is 1, i.e., the guidance from previous chunks is disabled, the score function in Eq. 9 simplifies to $\nabla _ { x _ { i } } \log \bar { p } _ { \mathrm { g u i d e d } } ( x _ { i } \mid x _ { < i } , \bar { c } _ { \mathrm { t e x t } } ) = ( 1 - w _ { \mathrm { t e x t } } ) \cdot \nabla _ { x _ { i } } \log p ( x _ { i } \mid$ $x _ { < i } ) + w _ { \mathrm { t e x t } } \cdot \nabla _ { x _ { i } } \log p ( x _ { i } \mid x _ { < i } , c _ { \mathrm { t e x t } } )$ . This form recovers the standard classifier-free guidance formulation widely adopted in bidirectional text-to-video diffusion models, which interpolates between unconditional and prompt-conditioned signals.

However, during our chunk-wise generation process, we observed subtle yet perceptible misalignments between adjacent chunks, resulting in temporal artifacts. This observation underscores the necessity of reinforcing temporal guidance to maintain chunk-to-chunk coherence. To this end, we increase $w _ { \mathrm { p r e v } }$ to 1.5, thereby amplifying the influence of preeding content. As shown in Fig. 7, this adjustment significantly enhances inter-chunk alignment and mitigates flickering artifacts, resulting in smoother and more temporally consistent video synthesis. Nevertheless, it should be noted that further increasing $w _ { \mathrm { p r e v } }$ beyond this optimal range may lead to saturation artifacts or even cause the video to become static (i.e., still frames) as playback progresses. We follow standard practice by setting $w _ { \mathrm { t e x t } }$ to 7.5.

![](images/8.jpg)  
Figure 8: Inference timestep sampling of non-distilled model.

# 2.4.2 Inference Timestep Sampler

Previous video generation studies have demonstrated that applying targeted timestep sampling strategies during inference can significantly improve generation quality. In our work, we observed similar behavior in MAGI-1. To enable finer-grained control over the sampling process, we introduce an additional tunable power transformation based on the scaling formula (Eq. 6) setting $w = 1 / 3$ and $k = 2$ yields the best visual quality, and visualization of the sampler is shown in Fig. 8.

# 2.4.3 Fine-Grained Control of Guidance Strength

Non-distilled Model. In the case of the non-distilled model, as described in Sec. 2.4.1, we set $w _ { \mathrm { p r e v } } = 1 . 5$ and $w _ { \mathrm { t e x t } } = 7 . 5$ during generation. In practice, when synthesizing longer videos (typically exceeding 5 seconds), we observe noticeable saturation and checkerboard artifacts progressively emerging during playback. These artifacts are primarily attributed to excessively strong guidance. However, uniformly reducing the strength of $w _ { \mathrm { p r e v } }$ and $w _ { \mathrm { t e x t } }$ often results in degraded content quality and increased flickering artifacts. This motivates a more fine-grained strategy in which the guidance scales are dynamically adjusted throughout the denoising process, that is, varying $w _ { \mathrm { p r e v } }$ and $w _ { \mathrm { t e x t } }$ as the denoising timestep $t$ progresses from 0 to 1.

To investigate when strong guidance is necessary, we analyze the evolution of latent representations throughout the denoising process (Fig. 6). As $t$ approaches 0.3, just before the final denoising stage begins, we observe that the decoded latent representations already exhibit coherent video content, with both structural and semantic elements largely established. The remaining denoising steps, from $t = 0 . 3$ to $t = 1 .$ primarily serve to refine local details, resembling a super-resolution process. Based on this observation, we hypothesize that strong guidance from either the text or previous chunks is no longer necessary during this stage. Accordingly, for $t > 0 . 3$ ,we reduce the guidance scales to $w _ { \mathrm { p r e v } } = 1 . 0$ and $w _ { \mathrm { t e x t } } \overset { \cdot } { = } 0 . 0 ,$ such that only the $\nabla _ { x _ { i } } \log { p ( x _ { i } \mid x _ { < i } ) }$ term remains active. As illustrated in Fig. 9, this simple yet effective adjustment significantly alleviates temporal artifacts and improves the overall coherence of longer video generations.

![](images/9.jpg)  
Figure 9: (a) When the generation length exceeds 5 seconds, severe artifacts emerge and intensify over time. (b) By adjusting the guidance strength (i.e., $w _ { \mathrm { p r e v } } = 1 . 0$ and $w _ { \mathrm { t e x t } } = 0 . 0$ when $t > 0 . 3$ ), there are no serious artifacts in the entire generation.

Distilled Model. A similar observation holds for the distilled model. Saturation artifacts progressively intensify as the video plays, motivating a comparable mitigation strategy. In the first three stages, we directly use the distilled model's output score, $\begin{array} { r } { \check { \nabla } _ { x _ { i } } \log p _ { \mathrm { d i s t i l l e d } } ( x _ { i } \mid } \end{array}$ $x _ { < i } , c _ { \mathrm { t e x t } } )$ . In the final denoising range, we incorporate additional guidance to reduce the influence of the previous chunk, even though the model has already undergone classifierfree guidance distillation. Specifically, we adopt the following guided score in the final stage:

$$
\begin{array} { r } { \nabla _ { x _ { i } } \log p _ { \mathrm { g u i d e d , d i s t i l l e d } } ( x _ { i } \mid c _ { \mathrm { t e x t } } , x _ { < i } ) = ( 1 - w _ { \mathrm { p r e v } } ) \cdot \nabla _ { x _ { i } } \log p _ { \mathrm { d i s t i l l e d } } ( x _ { i } \mid c _ { \mathrm { t e x t } } ) } \\ { + w _ { \mathrm { p r e v } } \cdot \nabla _ { x _ { i } } \log p _ { \mathrm { d i s t i l l e d } } ( x _ { i } \mid c _ { \mathrm { t e x t } } , x _ { < i } ) . } \end{array}
$$

This formulation is derived by switching the positions of $x _ { < i }$ and $c _ { \mathrm { t e x t } }$ in Eq. 8 and Eq. 9, resulting in the form $\begin{array} { r } { \nabla _ { x _ { i } } \log \bar { p } _ { \mathrm { g u i d e d } } ( x _ { i } \mid \check { c } _ { \mathrm { t e x t } } , \hat { x } _ { < i } ) = ( 1 - w _ { \mathrm { t e x t } } ) \cdot \nabla _ { x _ { i } } \log p ( \hat { x } _ { i } ) + ( w _ { \mathrm { t e x t } } - \check { c } _ { \mathrm { t e x t } } ) } \end{array}$ $w _ { \mathrm { p r e v } } ) \cdot \nabla _ { x _ { i } } \log p ( x _ { i } \mid c _ { \mathrm { t e x t } } ) + w _ { \mathrm { p r e v } } \cdot \nabla _ { x _ { i } } \log p ( x _ { i } \mid c _ { \mathrm { t e x t } } , x _ { < i } )$ By setting $w _ { \mathrm { t e x t } } = 1$ , thereby disabling the text guidance term, the first component vanishes and the expression simplifies to Eq. 10. The rationale behind this modification is that we do not introduce a nulltext token during distillation, and therefore do not explicitly model $p _ { \mathrm { d i s t i l l e d } } ( x _ { i } )$ or $p _ { \mathrm { d i s t i l l e d } } ( x _ { i } \mid x _ { < i } )$ In our experiments, we set $w _ { \mathrm { p r e v } } = 0 . 7 ,$ which effectively attenuates the influence of previous chunk guidance in the final denoising stage and helps mitigate temporal saturation artifacts.

# 2.4.4 KV Cache

Thanks to its auto-regressive nature, MAGI-1 can leverage the KV cache mechanism during inference, which is a widely adopted technique in language models to avoid redundant computations. Specifically, once a chunk has been sufficiently denoised, its features can be cached and reused by subsequent denoising chunks without the need for recomputation.

Furthermore, by constraining the KV range, MAGI-1 can easily support long video generation. For example, by setting the KV range to 8 for all chunks, each newly generated chunk depends only on the preceding 8 seconds of video content. This design ensures that the computational cost of generating long videos scales linearly with their duration.

In addition, many KV compression (Hooper et al., 2024; Xiao et al., 2023b; Sheng et al., 2023) techniques have recently been developed to reduce the computational overhead of auto-regressive model while preserving the ability to reference the full history as much as possible. MAGI-1 is theoretically compatible with these advancements, although we leave their exploration in MAGI-1 for future work.

MAGI-1 also benefits from the unique characteristics of denoising models: at higher noise levels, the model focuses on capturing global structural information, whereas at lower noise levels, it produces fine details and textures. By dynamically adjusting the KV range at different denoising stages, we can unlock new apabilities that were previously challenging o subject identities, or allowing changes in object identities while maintaining consistent global layouts. More details and experimental results are provided in Sec. 2.6.

# 2.5 Prompt-Enhancement Strategy

MAGI-1 is trained with highly descriptive captions that follow a specific structure as text conditions. However, in real-world scenarios, user inputs vary widely: ranging from very br prompts to overly elaborat desciptions. This mimatch between thetraig distribution and real user inputs often leads to suboptimal inference performance. To address this gap, we propose a Prompt Enhancement (PE) strategy during inference. We take the image-to-video (I2V) task as an example to illustrate our PE approach. In this setting, users typically provide an image along with an optional textual prompt. To enhance the user input, we employ a state-of-the-art multi-modal large language model (MLLM) to perform prompt refinement. Our PE pipeline consists of two parallel sub-processes:

The first sub-process analyzes and describes the content of the uploaded image. The second sub-process predicts the temporal evolution of the scene or objects in the first frame, such as actions, motion trajectories, and object transitions.

This structured enhancement strategy significantly improves generation quality. However, due to the large size of the state-of-the-art MLLM, it incurs high computational cost and latency, limiting its feasibility in real applications. To enable lightweight deployment, we distill the enhanced prompts generated by the large MLLM into a smaller, more efficient model (\~7B). We construct a training corpus of approximately 2 million examples, filtering out samples with excessively long target texts to ensure controlled output length. Based on human evaluation, the distilled model achieves comparable video generation quality to its larger counterpart, while greatly reducing inference latency and computational resource usage.

# 2.6 Model Capability Study

Real-time Streaming Video Generation The chunk-by-chunk pipelined inference of MAGI-1 offers two key advantages: (1) the time to display the first clear chunk is independent of the total generated video length; and (2) the generation latency between consecutive chunks is significantly reduced. Combined with a high-performance inference infrastructure, MAGI-1 enables real-time streaming video generation, unlocking new applications in interactive content and live streaming. More implementation details are in Sec. 4.2.1.

Chunk-wise Text Controllability Chunk-wise text controllability is one of the key features of MAGI-1, enabling us to decompose complex actions into simpler, shorter segments and significantly enhancing the model's ability to generate intricate action sequences. Furthermore, when combined with the capability of MAGI-1 for long video generation, this makes it possible to create videos with complex narrative structures, as illustrated in Fig. 10

Video Continuations Video continuation is a task that MAGI-1 natively supports. In the community, an alternative approach to video continuation relies on image-to-video generation (I2V), where the last frame of the given prefix video is used as the starting frame for the extended video. However, this approach often struggles to maintain consistent

![](images/10.jpg)

![](images/11.jpg)  
(a) A man smiles while resting his chin on his hand.

(b) He slowly rises from his seat.

![](images/12.jpg)

(c) Draws a pistol, from which a red rose is fired.

![](images/13.jpg)

(d) The rose transforms into a yellow bird that lands on his shoulder as he makes a playful expression.

![](images/14.jpg)

(e) He performs a juggling gesture as curtains on both sides gradually close, concealing him completely.

![](images/15.jpg)

(f) The curtains reopen by the man, then he turns and walks away.

![](images/16.jpg)  
(g) As he departs, a roaring lion logo slowly fades into view on the screen.   
Figure 10: This figure presents a near 30-second video generation example that demonstrates the capability of our model for complex actions and narrative structures through chunk-wise controllability and long-video generation. The sequence progresses from (a) to $( \mathbf { g } )$ ,with each sub-caption corresponding to the text prompt used during generation.

![](images/17.jpg)

(a) Text Guidance: A clear acrylic sheet placed on a wooden table with a small dollop of red paint.   
A rotating paintbrush attached to a rotating platform rotates clockwise and goes through the paint.   
Static shot with no camera movement.

![](images/18.jpg)

(b) Text Guidance: A grabber arm is holding a tennis bal above a piece of cardstock propped up on a rotating platform sitting on a table that rotates clockwise. The grabber lowers the ball and places is on the table as the cardstock rotates. Static shot with no camera movement.

Figure 11: Comparison between video-conditioned (V2V) and image-conditioned (I2V) video continuation. (a) MAGI-1 (V2V) accurately captures the pen's rotational trajectory by leveraging historical motion information, while I2V fails to reproduce the correct motion due to the absence of temporal context." (b) In an occlusion scenario, V2V successfully predicts post-occlusion behavior by utilizing information before the occlusion, whereas I2V shows poor temporal consistency. Each example presents the real-world scene (top row), MAGI-1 (V2V) generation (middle row), and MAGI-1 (I2V) generation (bottom row).

motion trajectories between the generated continuation and the prefix video, leading to motion discontinuities or generating implausible predictions due to the loss of essential historical information. Fig. 11 shows such cases. In the pen rotation example, I2V fails to capture the correct rotational velocity because it lacks access to preceding motion dynamics. Similarly, in the occlusion scenario, 12V cannot accurately predict the object's reappearance after occlusion due to missing temporal information. In contrast, conditioning on the full prefix video allows MAGI-1 to naturally preserve motion continuity by leveraging historical patterns and temporal dependencies, enabling seamless video continuation.

Controllable Shot Transition Another exciting feature of MAGI-1 is its ability to enable diverse and controllable transitions at any designated chunk by adjusting the KV range across different denoising stages. Specifically, by setting the KV range to 1 only at high-noise denoising stages (meaning the model cannot access the preceding video content) while keeping a normal KV range (e.g., 8) at other stages, we can achieve shot transitions while preserve object identities unchanged, as shown in Fig. 12a. Conversely, by setting the KV range to 1 only at low-noise stages, we can produce transitions where the overall layout of the scene remains consistent, but the fine details of the objects change, as illustrated in Fig. 12b.

We believe the above capabilities can offer an entirely new level of creative control for video content creation.

![](images/19.jpg)  
(b) Transition with consistent scene layout but changing object details.   
Figure 12: This figure illustrates two examples of realizing distinct shot transitions by modulating the KV range at different denoising stages. (a) demonstrates a case where the KV range is set to 1 only at the high-noise denoising stages, whereas (b) applies it at the low-noise denoising stages.

# 3 DATA

Training a high-performance video generation model demands massive, high-quality, and diverse data. To this end, we have developed a scalable data processing system that constructs the training dataset for MAGI-1 from tens of petabytes of raw videos and images collected from a wide range of sources.

An overview of the data processing pipeline is shown in Fig. 13. We utilize PySceneDetect5 to cut long videos into short clips, ensuring that each clip contains only a single shot. Next, we apply a series of filters to remove low-quality data and eliminate duplicates. While this initial filtering stage effectively discards most of the low-quality data, some problematic cases still persist. To further improve data quality, we incorporate a multi-modal large language model (MLLM) as a stronger filter. Data that passes this filter is then captioned by the MLLM to provide accurate and detailed descriptions.

Through this process, we curate training data with satisfactory visual and motion quality. However, the distribution of the data — particularly in terms of semantic concept — still requires consideration. Specifically, we observed that the modeling difficulty varies significantly across different concepts. To address this, we use a dynamic distribution adjustment strategy based on evaluation results obtained during training. Additionally, we tailor the data distribution to accommodate the multi-stage training strategy.

In the sections that follow, we provide a detailed description of each component in our data processing pipeline.

![](images/20.jpg)  
Figure 13: Overview of the our data processing pipeline. The shot cutting module is only applied for video data.

# 3.1 Filter Actors

We have developed a set of filtering actors to ensure the quality of the training data. These actors are described below in details:

Video Quality Assessment We adopt DOVER (Wu et al., 2023) to assess the visual quality of each video clip. DOVER provides three distinct quality scores: overall score, aesthetic score, and technical score. Through empirical evaluation, we found that the technical score alone is the most effective indicator for our use case.

Aesthetics We employ the LAION aesthetic model (Schuhmann et al., 2022) to predict aesthetic score for each image and video. Since the LAION aesthetic model is originally designed for images, we use the aesthetic score of the first frame to represent the quality of the entire video clip.

Overexposed and Underexposed Some videos suffer from overexposure or underexposure, which we have found to adversely affect training stability. To remove such data, we convert every frame of the video to the HSI color space and compute the average brightness across the entire video. Videos identified as either overexposed or underexposed, based on their average brightness, are excluded from the training set.

Motion Strength To quantify the motion strength of each video, we employ the RAFT optical flow model (Teed & Deng, 2020). To reduce computational overhead, all videos are first downsampled to 8 FPS before computing the optical flow between adjacent frames. The optical flow is calculated at the pixel level, and the overall motion strength is obtained by averaging the flow magnitudes across all pixels in the clip.

However, this approach tends to underestimate motion in cases where the background remains static while the foreground exhibits significant movement. To mitigate this issue, we additionally apply a saliency detection model (Zhao & Wu, 2019) to each frame. The resulting saliency maps enable us to distinguish between foreground and background regions, allowing us to compute the average optical flow separately for both.

As a result, we derive three motion statistics: overall motion strength, foreground motion strength, and background motion strength. To balance data quality and training difficulty, we prioritize video clips with moderate motion strength, avoiding both overly static and excessively dynamic videos. Specifically, we define lower and upper thresholds for all three motion statistics to guide data selection.

Camera Movement Stability A significant portion of collected videos is captured with handheld devices, which often results in erratic camera movements that are challenging for the model to learn. Since such cases are not effectively filtered by motion strength alone, we estimate camera stability by evaluating the consistency of optical flow between adjacent frames, filtering out clips with unstable camera motion.

Slides Movement Slide movements, such as floating photos or banners commonly found in screen recordings or slideshow presentations, are another undesirable case. To detect these, we analyze the divergence of the optical flow across all pixels in each frame. If the divergence remains consistently low over time, the clip is identified as containing slide movements and is removed.

Border Detection We perform edge detection on each frame and apply the Hough transform to identify persistent vertical and horizontal lines across frames. These lines are treated as potential borders, and the proportion of frames containing such borders serves as a confidence score for filtering.

Text Detection We perform text detection on video frames to identify and exclude clips containing excessive textual content. Specifically, if any frame within a clip contains an overly large number of characters or if the detected text regions occupy a substantial portion of the frame, the corresponding clip is discarded.

A notable exception is subtitles, which typically consist of fewer characters and occupy relatively limited spatial regions, rendering them less likely to be filtered out by the aforementioned criteria. Nevertheless, subtitles exhibit distinctive spatiotemporal patterns: they consistently appear in fixed locations where most commonly at the top or bottom of the frame, and persist across multiple consecutive frames. By leveraging these characteristics, we are able to reliably detect and exclude video clips containing subtitles from the training data.

Logo Detection Many videos contain logos in the corners, which is an undesirable pattern for model training. To address this, we employ the Florence-2 model (Xiao et al., 2024), which supports open-vocabulary object detection. By providing a predefined set of keywords, Florence-2 accurately detects and localizes logos within video frames and providing confidence scores for filteing.

Corner Face Detection In commentary videos, narrators typically appear in a fixed corner of the screen, and we aim to exclude such patterns from our training data. To achieve this, we employ a face detection model, leveraging both face location and detection confidence to identify potential narrators. Specifically, we average the detection confidence of faces located in fixed corners across all frames to estimate the likelihood of a narrator's presence.

Transition Detection While PySceneDetect can segment raw videos into clips based on shot boundaries, it struggles to handle complex transitions, and as a result, the resulting clips may still contain multiple shots. To address this issue, we sparsely sample keyframes from each video and use CLIP (Radford et al., 2021) to compute the semantic similarity between adjacent keyframes. If the similarity falls below a predefined threshold, the clip is considered to contain multiple shots and is subsequently removed.

# 3.2 De-duplication

Recent studies on large language models (Lee et al., 2021; Hernandez et al., 2022) have shown that even small amounts of duplicate data can significantly degrade performance. Motivated by this, we conduct rigorous de-duplication. We compute pairwise similarity usi both LIP (Radord et al., 2021) and DINOv2 (Oquab et al., 2023), and tret ay clip exceeding the threshold in either similarity as a duplicate to be removed.

# 3.3 MLLM as Advanced Filter

After the above filtering and de-duplication processes, most of the undesired data have been effectively removed. However, due to the limitations of the current filtering actors, a small portion of low-quality data still remains. As the remaining data size has been significantly reduced and to further improve data quality, we leverage a multi-modal large language model (MLLM) to perform an additional round of filtering. This enables us to detect more complex bad cases. Notably, this step can be seamlessly integrated into the subsequent caption procedure, thereby reducing overall costs and improving efficiency.

# 3.4 Caption

Highly Descriptive Caption Recent advances (Betker et al., 2023) have demonstrated that using MLLMs to generate highly descriptive captions is crucial for improving image generation quality, and we adopt this approach for captioning our data." Compared to images, videos have richer temporal information, including actions, camera movements, and scene changes. However, most mainstream MLLMs are primarily designed for images. T sequence. Through empirical analysis, we find that using 4 to 12 frames per video clip (depending on its duration) reaches the best trade-off between descriptive accuracy and computational efficiency. For video data, the captioning prompt is structured into two stages. In the first stage, the model is guided through a series of targeted questions aimed at eliciting responses on predefined attributes of the video clip (as summarized in Tab. 3). This step encourages the model to perform a structured analysis of the content. In the second stage, the model generates the final descriptive caption, which can incorporate salient observations identified in the preceding analysis of first stage. In contrast, for image data, we directly prompt the model to generate a caption without the attribute-based pre-analysis. Example captions are provided in Tab. 4.

Table 3: Predefined attributes used in caption instruction.   

<table><tr><td>Attribute</td><td>Instruction</td></tr><tr><td>Scene Count Camera Transitions Camera Shot Type Camera Movement Main Subject Identification Subject Attributes</td><td>Identify the number of distinct scenes in the video. Note any noticeable transitions between shots. Specify the type of camera shot used. Describe any camera movements. Determine who or what is the central focus of the video. Describe the main subject&#x27;s appearance.</td></tr></table>

Auto-Regressive Caption Unlike typical bi-directional denoising video generation models that produce an entire video as a whole, our model generates videos in an auto-regressive manner. This design allows our model to condition different parts of the video on distinct text prompts, offering greater controllability. To enable this capability, we provide fine-grained, second-by-second descriptions for each video clip. Tab. 4 shows example. Specifically, the caption of the first second is instruct to generates a detailed description. For caption of subsequent seconds, they focus on describing changes relative to the previous one.

# 3.5 Data Adjustment in Training

We have two different data adjustment scenarios during training. First, we use a multi-stage training strategy, with later stages having higher data quality; Second, we dynamically adjust the data distribution during training based on the evaluation results.

Multi-stage Adjustment MAGI-1 is trained in three stages, with the data resolution gradually increasing from 256p to $4 8 0 \mathrm { p }$ and ultimately to $7 \hat { 2 } 0 \hat { \mathrm { p } }$ . Alongside the resolution improvements, the data volume is progressively reduced, and more rigorous filtering strategies are employed to ensure higher data quality. Furthermore, in the final stage, the video duration is extended from a maximum of 8 seconds to a maximum of 16 seconds, allowing the model to capture richer temporal dynamics. The data specifications for each stage are summarized in Tab. 5.

Dynamic Distribution Adjustment An appropriate data distribution is crucial for training high-performance models. However, identifying the optimal distribution in advance is challenging. For instance, during training, we observed that landscape scenes are relatively easy for the model to learn, while human expressions are significantly more difficult. These insights are hard to predict beforehand. To address this, we adopt a dynamic distribution adjustent statey. By continuousy monitoring mod perforan throghout the trainng process, we can adaptively adjust the proportion of specific data subsets to strengthen the underperforming aspects of the model, thereby enabling a more effective learning process.

# 4 Infrastructure

In this section, we introduce our training infrastructure and inference infrastructure.

Table 4: Caption examples used in MAGI-1.   

<table><tr><td>Caption Type Video Detail Caption</td><td>Example Medium shot of a hotel reception desk with two staff members. A</td></tr><tr><td rowspan="2"></td><td>woman stands on the left, and a man in a suit and red tie stands on the right. White orchids are in vases on either side of the desk. A painting hangs on the wall behind the desk. The man on the right picks up a telephone receiver and begins a phone conversation. The man is now more prominently featured in the frame, his up-</td></tr><tr><td>per body taking up a larger portion of the screen. The woman on the left is still visible, but less prominent. The man continues his phone conversation, his expression becoming more serious. The new arrival is now standing at the reception desk, slightly</td></tr><tr><td>Image Detail Caption</td><td>the frame. A young woman with long dark hair stands on a rocky beach. She is wearing a light beige, strapless top and matching wide-legged pants. Her arms are crossed, and her hands are near her chest. She is barefoot. The rocks are various shades of brown and tan, some smooth and some rough. The rocks are wet in places. The ocean is visible in the background. The sky is light blue and mostly clear. A small child is partially visible in the lower left corner of the frame, seemingly playing near the water&#x27;s edge. The woman is positioned in the center of the frame, slightly off-center towards the right. She is facing the camera directly. The rocks behind her are large and form a backdrop to her figure. The rocks in the foreground are smaller and scattered around her feet. The child is in the lower left corner, facing towards the center of the frame, and is partially obscured</td></tr><tr><td rowspan="2">AR Caption</td><td>elements of the beach. 1st second: A woman holds a lipstick tube, her expression changes subtly. The background is a simple, light brown wooden wall. The woman in the frames is wearing a beige lace sleeveless top and gold necklaces. She holds a gold lipstick tube in her right hand. Her makeup is subtle, and her expression changes slightly throughout the two frames. Her hair is dark brown and styled in a shoulder-length cut. The lighting is soft and even, creating a neutral mood. There are no other objects or people visible in the frames.</td></tr><tr><td>2nd second: The woman&#x27;s head tilts slightly, her expression shifts from a neutral to a slight smile. The lipstick remains in her hand. The camera remains static, focusing on the woman. 3rd second: The woman&#x27;s head is slightly turned to the left, her expression is more serious. The lipstick is still in her hand. The camera</td></tr></table>

Table 5: Data configuration of different stages.   

<table><tr><td></td><td>stage-1</td><td>stage-2</td><td>stage-3</td></tr><tr><td>Resolutions</td><td>256p/360p</td><td>480p</td><td>720p</td></tr><tr><td>Video Duration</td><td>≤ 8s</td><td>≤ 8s</td><td>≤ 16s</td></tr><tr><td>Image-Video Ratio</td><td>4:1</td><td>4:1</td><td>4:1</td></tr><tr><td>AR Caption Ratio</td><td>0%</td><td>10%</td><td>10%</td></tr></table>

# 4.1 Training Infrastructure

Efficient training of large-scale autoregressive denoising models like MAGI-1 requires carefully tailored distributed training infrastructure. Existing distributed training frameworks, such as Megatron (Shoeybi et al., 2020) and DeepSpeed (Rajbhandari et al., 2020) are primarily designed for large language models (LLMs). However, MAGI-1 differs significantly from LLMs in both algorithmic side and data side.

On the algorithmic side, MAGI-1 integrates both autoregressive and denoising modeling paradigms, resulting in a model architecture that is notably more complex than that of typical LLMs. It incorporates components such as gating, cross-attention, and block-causal attention that are rarely used in language models.

On the data side, a single video training example typically contains tens to hundreds of times more tokens than a text example. Furthermore, ensuring the temporal and semantic integrity of video content imposes strict constraints, making it infeasible to directly apply common data processing strategies from LLMs, such as arbitrary sequence truncation or concatenation of multiple samples into a single training sequence offline, in the context of video generation.

These fundamental differences introduce unique challenges, necessitating a new, purposebuilt distributed system design. In this section, we propose novel solutions to address these challenges to enable efficient and scalable training of MAGI-1.

Specifically, the training of MAGI-1 leverages a combination of data parallelism (DP), context parallelism (CP), and tensor parallelism (TP). To address the DP load imbalance caused by variable-length video sequence and the insufficient GPU utilization on short token sequences, we introduce a distributed Packing and Padding $( \mathrm { P n P } )$ during training, that performs online batching of video data in each training iteration. This strategy mitigates GPU bubbles thereby significantly improving overall training efficiency (Sec. 4.1.1).

Due to the use of $\mathrm { P n P }$ and the inherent demands of block-causal attention in MAGI-1, we require an efficient attention implementation capable of supporting highly flexible attention masks. Additionally, given the extremely long token sequences typical in video training data, native support for context parallelism is essential. To address these requirements, we propose MagiAttention: a scalable distributed attention mechanism that efficiently handles diverse attention masks and is optimized for ultra-long sequences (Sec. 4.1.2).

Through the above innovations, we enable the efficient training of MAGI-1. However, while developing the MAGI-1 training system, we identified several limitations in existing large-scale training frameworks. For instance, most current frameworks (including ours) do not treat verifiable numerical accuracy in distributed environments as a first-class design concern. Moreover, the tight coupling between algorithm development and infrastructure implementation often creates friction between algorithm researchers and infrastructure engineers, hindering efficient collaboration. To address these challenges, we discuss potential directions and design principles for next-generation training infrastructure in Sec. 4.1.3, with the goal of providing insights and practical guidance for the broader research and engineering community.

# 4.1.1 Distributed Packing and Padding

Due to the integrity constraints of video data and the variability in video lengths and resolutions, we adopt a Packing and Padding $( \mathrm { P n P } )$ strategy (Sirluk, 2024; Kundu et al., 2024) to batch video samples in a way that minimizes excessive padding and reduces unnecessary computational overhead in distributed training scenarios. Moreover, the data composition is frequently adjusted during the training of MAGI-1 (See Sec. 3.5), and to accommodate such flexibility, we employ an online $\mathrm { P n P }$ strategy instead of a offline approach.

The core idea of $\mathrm { P n P }$ is to efficiently utilize GPU resources by concatenating multiple short sequences into a batch while minimizing redundant filling. The offline formulation of this problem aligns with the classic bin-packing problem: given a set of input samples, the goal is to pack them into a set of bins, each with a fixed capacity max_length, while minimizing overall unused space. Although this problem is NP-complete, it can be efficiently approximated in practice using the First-Fit Decreasing (FFD) (Dósa, 2007) greedy algorithm.

In our online setting, we must process streaming data inputs while ensuring compatibility with the 3D parallelism strategy employed during training. To this end, we reformulate the problem as follows: given M candidate samples, we aim to pack them into N bins of size max_length, minimizing overall space waste. Here, M denotes the size of the candidate pool with $\mathbf { M } \gg \mathbf { N } .$ ; N must be divisible by the DP_SIZE; and max_length must be divisible by $\mathsf { T P \_ S I Z E } \times { \mathsf { C P \_ S I Z E } }$ .

In practice, we extend the FFD algorithm with custom heuristics to support efficient online packing under these constraints. This approach enables us to achieve a $9 9 \%$ capacity utilization rate and the differences between different DP groups can be neglected, thus substantially reducing computational overhead during training.

# 1.1.2 MagiAttention: Towards Linear Scalability for Ultra-Long and Heterogeneous Mask Training.

Training large-scale autoregressive diffusion models like MAGI-1 for video generation presents two major challenges:

•The extremely long context length of video tokens, which reaching up to 4 million during training, results in prohibitive computational and memory overhead. ContextParallelism (CP) is designed for dealing such long context challenge, but existing state-of-the-art P methods (Jacobs et al., 2023; Liu et al., 2023; Fang & Zhao, 2024; Gu et al., 2024; Chen et al., 2024b) face scalability limitations that face scalability limitations due to size constraints or the high communication overhead inherent in inefficient ring-style point-to-point (P2P) patterns. While recent efforts (Wang et al., 2024; Zhang et al., 2024; Ge et al., 2025) dynamically adjust CP sizes to avoid unnecessary sharding and redundant communication for shorter sequences, they still incur extra memory overhead for multiple NCCL process groups and involve complex scheduling to balance loads and synchronize across different subsets of ranks. •The combination of block-causal attention and Packing-and-Padding introduces highly complex attention mask patterns (Sec.4.1.1), which cannot be efficiently handled by existing attention implementations.

To address the aforementioned challenges, we propose MagiAttention, which aims to support a wide variety of attention mask types (i.e., kernel flexibility) while achieving linear scalability with respect to context-parallel (CP) size across a broad range of scenarios. Achieving this goal depends on meeting the following fundamental conditions:

•Linearly Scalable Attention Kernel: The performance of the attention kernel should not degradate as CP size increases. To this end, we introduce Flex-Flash-Attention, an extension of FlashAttention-3 (FA3), which native considers the efficiency impact of attention mask partitioning in distributed environments. It supports distributable mask representations with a tailored kernel implementation to ensure scalability while accommodating a broader range of attention mask types.

![](images/21.jpg)  
Figure 14: Overview of MagiAttention: (1) Flex-Flash-Attention(FFA), an efficient attention supports flexible mask patterns and native considers distribution requirements; (2) The dispatch solver shards and dispatches packed data with ultra-long contexts and heterogeneous masks, ensuring load-balanced computation; (3) Group-Cast and Group-Reduce primitives eliminate redundant communication; (4) The adaptive multi-stage overlap strategy effectively hides communication latency; (5) Forward and backward timelines of MagiAttention. With all techniques together, MagiAttention reach linear scalability under diverse scenarios.

•Balanced Computational Workloads: Imbalances in the computational load across CP ranks lead to unavoidable idle bubbles that hinder scalability. MagiAttention is natively designed to ensure Computation Load Balancing, mitigating such inefficiencies. Full Overlap of Communication and Computation: Without sufficient overlap, increasing CP size results in communication-induced idle time on GPUs, impairing scalability. MagiAttention introduces novel Zero-Redundant Communication Primitives to minimize communication overhead, along with an Adaptive Multi-Stage Overlap strategy that enables effective communication-computation overlap.

The overview of MagiAttention is shown in Fig. 14, and we will introduce key designs in the following, with comprehensive experimental results presented in Appendix B.2.

Flex-Flash-Attention. FlashAttention (Dao et al., 2022; Dao, 2023; Shah et al., 2024) is a foundational technique in large-scale model training for its superior performance and support for varlen-packed data with causal attention masks. However, it offers limited support for irregular attention masks, particularly when such patterns are distributed across CP ranks, resulting in increased complexity and underscoring the need for a more flexible attention kernel (PyTorch; Dong et al., 2024; Wang et al., 2025b) without compromising performance.

Therefore, we introduce Flex-Flash-Attention (FFA), which is natively designed for distribution scenarios and provides greater flexibility in handling diverse attention mask types. The core idea behind FFA is to generalize a distributable formulation for irregular attention masks by decomposing the entire mask into multiple computational units, each referred to as an AttnSlice. Each AttnSlice is defined by a triplet QRange, KRange, MaskType, which specifies a submask with a basic shape bounded by a contiguous 2D query-key region (see Fig. 20). Using this formulation, a wide variety of commonly used attention masks (Fig. 15)

![](images/22.jpg)  
Figure 15: Examples of mask patterns formulated by AttnSlice. (a)-(d) Standard FA3- compatible patterns; (e)-(h) Irregular masks beyond FÅ3's capabilities, including our novel varlen block-causal design, which FFA supports seamlessly while maintaining performance comparable to FA3.

(including our varlen block-causal mask) can be expressed as a composition of multiple such triplets, making FFA highly suitable for distributed attention computation.

Built on FA3 kernels, Flex-Flash-Attention leverages NVIDIA Hopper GPUs' TMA feature (NVIDIA, 2024) and introduces slice-level parallelism with atomic operations for correctness (Fig 21), achieving comparable MFU to FA3 while supporting the flexible AttnSlice formulation 6 (see Appendix B.2 for benchmarks).

Computation Load-Balance. In context-parallelism (CP) settings, different CP ranks may be assigned heterogeneous attention masks, resulting in imbalanced computational workloads across ranks. Ring-Attention (zhuzilin, 2024) employs a specialized partitioning strategy designed specifically for causal attention, which limits its applicability to more general attention patterns. To overcome this limitation, we propose a generic and efficient dispatch solver that enables balanced workload distribution across CP ranks for a broad range of attention types.

(Fig 14 (2)). Specifically, the entire mask is evenly partitioned along the query-dimension int dispatch chunks, each associated with a submask ar: $\{ ( C _ { i } , \bar { \mathrm { A r e a } } ( \bar { C _ { i } } ) ) \} _ { i = 1 } ^ { \bar { n } } ,$ where $C _ { i }$ indicateshpatc hu, $\mathrm { A r e a } ( C _ { i } )$ is the mask area of $C _ { i } , n$ is $\frac { s e q l e n } { d i s p a t c h \_ c h u n k \_ s i z e } ,$ and dispatch_chunk_size is a hyperparameter controlling granularity. These dispatch chunks are then equally assigned to $c p .$ _size buckets, with each bucket containing the exact same number of dispatch chunks to ensure token-level load balance in non-attention modules, a $\left\{ \left( B _ { j } , \mathsf { S u m A r e a } ( B _ { j } ) \right) \right\} _ { j = 1 } ^ { c p \_ s i z e }$

With above strategy, we could fine-grained control the computational workloads of each CP rank, and the load-balancing dispatch becomes a combinatorial optimization problem, ine $f ^ { * } : \{ C _ { i } \} _ { i = 1 } ^ { n }  \{ B _ { j } \} _ { j = 1 } ^ { c p _ { - } s i z e }$ as folows

$$
\begin{array} { l } { { f ^ { * } = \displaystyle \arg \operatorname* { m i n } _ { f } \operatorname* { m a x } _ { j } \left\{ \mathrm { S u m A r e a } ( B _ { j } ) \right\} } } \\ { { \mathrm { s . t . } ~ | B _ { j } | = \displaystyle \frac { n } { c p _ { - } s i z e } , ~ s e q l e n ~ ^ { \% } ~ ( c p _ { - } s i z e \times d i s p a t c h \_ c h u n k \_ s i z e ) = 0 } } \end{array}
$$

However, this optimization is a known NP-hard problem, making it impractical to find an optimal solution on-the-fly during each training iteration, especially given the varying mask patterns across micro-batches. Thus, we propose an efficient greedy algorithm (as shown in Alg. 1) that provides a suboptimal yet effective solution within ${ \cal O } \breve { ( } n \log n )$ complexity.

Zero-Redundant Communication Primitives. The existing ring-style implementation uses point-to-point send/recv communication primitives, which cannot provide sufficient communication granularity, resulting in redundant communication. Take causal mask as an example, we analyze the redundant communication by recording the distribution of remote key-value (KV) requests and their gradients (dKV) under sparse attention masks. As shown in Fig 23, $\mathrm { K V _ { 0 } }$ is required by all queries and should be sent to all devices via Broad-Cast in the forward pass, with $\mathrm { d K V } _ { 0 }$ reduced via All-Reduce in the backward pass. In contrast, $\mathrm { K V } _ { 7 }$ is only needed by its host device but still circulates through all devices, and this redundancy intensifies in varlen scenarios.

To address this, we introduce two communication primitives: group-cast and group-reduce, which model the communication patterns of low-demand KV and dKV (Fig 24). For example, in the causal mask, $\mathrm { K V } _ { 5 }$ on rank2 is required only by $\{ \mathrm { Q } _ { 6 } , \mathrm { Q } _ { 7 } \}$ and should be sent exclusively to the target ranks $\{ { \mathrm { r a n k } } _ { 0 } , { \mathrm { r a n k } } _ { 1 } \}$ via group-cast, while the partial $\mathrm { d K V } _ { 5 }$ is collected and reduced back to rank2 via group-reduce accordingly.

As no existing communication kernels support these primitives, we prototype them using all-to-all- $\cdot \mathsf { v }$ (Fig 24), achieving zero-redundant communication in both forward and backward passes. However, this approach introduces extra pre-/post-processing overhead, similar to (un)permutation in expert parallelism (EP) (Gale et al., 2022). While kernel fusion mitigates the overhead, a dedicated implementation of group-cast and group-reduce remains a key direction for future work.

Adaptive Multi-Stage Overlap. Leveraging previous optimizations, we achieve highperformance computation through an efficient kernel and balanced workload dispatch, while minimizing communication overhead with our new primitives. To drive true linear scalability, we further improve end-to-end performance by introducing a multi-stage compute-communication overlap strategy, that effectively hides communication latency and adaptively optimizes overlap through manual or automatic tuning.

Similar to prior works (Liu et al., 2023; Zhao et al., 2023; He et al., 2024), we schedule pipeline stages to overlap computation with communication for both forward and backward passes (Fig 25). Each rank; first partitions its remote KV /dKV communication into stages. In the forward pass, the scheduler first launches the group-cast kernel to prefetch the next remote KV, then asynchronously executes the FFA kernel for partial attention computation, hiding all communication behind computation 7. In the backward pass, besides prefetching the next KV, the group-reduce kernel reduces the last dKV in a separate CUDA stream before launching the FFA kernel for the current stage, ensuring communication is overlapped across all stages except the final dKV reduction 8.

To adaptively control overlap granularity, we further introduce a tunable hyperparameter, num_stages, accounting for varying compute-to-communication ratios across training setups, microbatches, or between forward and backward passes. This parameter can be manually configured or automatically determined by our overlap solver, with a simple dynamic search algorithm (See Alg. 2 for more details).

# 4.1.3 Rethinking System Design for Robust Distributed Training Frameworks with DTensor

As large-scale models continue to evolve, the growing complexity of training procedures has exposed fundamental limitations in existing distributed training frameworks (Shoeybi et al., 2020; Rajbhandari et al., 2020). Two major bottlenecks are particularly prominent:

Lack of testability by design. Most frameworks were not initially built with testability as a first-class feature, resulting in fragile infrastructure with limited maintainability and reliability; • Tight coupling between model implementation and parallelization strategy. This entanglement prevents algorithm researchers and system engineers from working independently, hindering collaboration and modular development

We argue that next-generation distributed training frameworks must directly address these two pain points to support large-scale model research and deployment.

Inspired by early explorations (Xu et al., 2021; Yuan et al., 2022) and PyTorch's pioneering implementations (Zhao et al., 2023; Team, 2024b; Liang et al., 2024), we propose a blueprint for redesigning robust distributed training frameworks based on Pytorch Distributed Tensor (DTensor) (Team, 2024b) and Parallel Plan:

DTensor PyTorch DTensor introduces three parallel placements: Replicated, Shard, and Partial, alongside a distributed initialization strategy to maintain placement semantics (Contributors, 2025a), and a propagation mechanism that deduces output placements from input ones for supported ops, triggering communication as needed 9 (Contributors, 2025b). While it supports basic ops including naive distributed matmul, its current implementations lack the generality to handle more complex yet commonly scenarios in modern training workflows, as shown in Tab. 12.

Parallel Plan Parallel Plan provides a declarative interface for specifying parallelization strategies across model submodules. It works in conjunction with the parallelize_module function and is built on top of DTensor. However, its current capabilities are mostly limited to tensor parallelism (TP) and do not generalize well to other parallelism.

In our architecture design, we extend both DTensor and Parallel Plan to support a broader range of usages. These extensions enable the following key features:

Decoupling Modeling from Parallelization. This feature allows model researchers to concentrate on model design and algorithm development without needing to manage low-level parallelism details. At the same time, infrastructure engineers can independently optimize parallelization strategies without modifying model implementation. This clear separation of concerns enables more efficient collaboration and improved training throughput.

High-Precision Alignment with Non-Distributed Oracles. By disabling all parallel plans, we can seamlessly revert to non-distributed configurations, yielding "pure" model code that serves as a baseline or oracle for evaluating distributed correctness. To ensure alignment within a relative error of $1 0 ^ { - 8 }$ , we upcast tensors to higher precision 10, enforce deterministic algorithms (Team, 2024a), and control randomness using consistent seed management. This design enables precise infrastructure testing, ultimately improving reliability and debuggability.

# 4.2 Inference Infrastructure

As an innovative large-scale autoregressive denoising model, MAGI-1 introduces two pivotal architectural innovations: multi-chunk parallel inference and KV cache, which unlock new possibilities for user experiences, such as real-time streaming video generation, and enables cost-effective deployment. However, these advancements also introduce new challenges to the inference infrastructure. In this section, we present our infrastructure design tailored to two major scenarios: real-time streaming inference on H100/H800, and cost-efficient deployment on RTX 4090 GPU.

# 4.2.1 Real-Time Streaming Video Generation

Our model adopts an auto-regressive architecture that supports real-time streaming video generation. To ensure a seamless user experience, we optimize for two key latency metrics: Time to First Chunk (TTFC), which measures the delay between task submission and starting to see the video, and Time Per Output Chunk (TPOC), which reflects the time required to generate each subsequent chunk. Maintaining a low TTFC enhances responsiveness, while keeping TPOC below 1 second is essential for uninterrupted playback.

We encountered three major challenges when designing the infrastructure:

• MAGI-1 consists of multiple sub-models: T5 for text embedding extraction, a VAE encoder for processing user-uploaded images and prefix videos, a VAE decoder for decode the denoised output, and a core auto-regressive denoising model. These components exhibit distinct computational characteristics: T5 and VAE are memory-bound, while the denoising model is compute-bound. Efficiently handling this heterogeneity is essential.   
To meet the TPOC target of under 1 second, MAGI-1 demands approximately 9 PFLOPS of compute per second of video, which far exceeds the capabilities of a single H100/H800 GPU. Achieving this requires serving models on multiple H100/H800 GPUs and a highly optimized parallelism strategy. First-chunk inference differs significantly from subsequent chunks. It is not computebound but CPU-bound, due to limited token workloads per GPU, resulting in a long TTFC.

To address these challenges, we propose a systematically optimized framework, enabling real-time streaming video generation for our largest 24B MAGI-1 model on 3-node, 24 H100 GPUs. Here, we briefly introduce our solutions.

Multi-Model Heterogeneous Serving Pipeline We designed a heterogeneous serving the VAE to cost-efficient hardware. This approach enables concurrent execution of MAGI-1 inference and VAE decoding, minimizing idle time and improving overall throughput. Profiling-driven resource allocation strategies further enhance utilization efficiency. With this design, we could efficiently handling the heterogeneity of different models and achieve the best performance.

TPOC Optimization Given that the denoising model of MAGI-1 is compute-bound, we prioritized aggressive quantization and distributed inference optimizations:

•Quantization. We adopted W8A8 SmoothQuant Xiao et al. (2023a) to quantize both weights and activations to FP8 precision, except the first and last layers. The quantization delivered a $3 0 \%$ speedup without compromising generation quality. Multi-Node Parallel Inference. We adopt a Ulysses-based multi-node parallel inference strategy with sufficiently computation and communication overlapping (less than $3 \%$ of communication time remaining unoverlapped in the execution timeline). As a result, the TPOC is optimized to be within 1 second when we generating $4 8 0 \mathrm { p }$ (3:4 aspect ratio) videos using 16 denoising steps and KV range of 5 on 24 H100/H800 GPUS.

TTFC Optimization For first-chunk inference, only a few hundred tokens need to be processed. In this scenario, the GPU workload is relatively light, and CPU-side bottlenecks become the primary constraint. To address this issue, we employ CUDA Graphs to minimize kernel launch overhead, reducing $3 0 . 4 \%$ latency. Additionally, we accelerate VAE decoding through a tile-based parallel mechanism and torch . compile, bringing latency down from 1 second to around 70 milliseconds. Collectively, these optimizations reduced TTFC to 2.3 seconds, ensuring a smooth real-time streaming experience. Tab. 6 summarizes the key optimizations and their corresponding latency gains11

Table 6: Inference Optimization and Latency Gain   

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Optimization</td><td rowspan=1 colspan=1>TTFC(s)</td><td rowspan=1 colspan=1>Gain</td><td rowspan=1 colspan=1>TPOC(s)</td><td rowspan=1 colspan=1>Gain</td></tr><tr><td rowspan=3 colspan=1>AutoregressiveDenoising Model</td><td rowspan=3 colspan=1>BaselineKV CacheUlyssesSmooth QuantCuda Graph</td><td rowspan=3 colspan=1>73.3473.343.863.002.30</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>45.49</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>23.94</td><td rowspan=2 colspan=1>1.90X18.0X1.29X-</td></tr><tr><td rowspan=1 colspan=1>18.0X1.29X1.30X</td><td rowspan=1 colspan=1>1.260.980.98</td></tr><tr><td rowspan=1 colspan=1>Vae Decoder</td><td rowspan=1 colspan=1>BaselineTile Paralleltorch.compile</td><td rowspan=1 colspan=1>1.000.200.07</td><td rowspan=1 colspan=1>5.00X2.86X</td><td rowspan=1 colspan=1>1.000.200.07</td><td rowspan=1 colspan=1>5.00X2.86X</td></tr><tr><td rowspan=1 colspan=1>End-to-End</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.37</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1></td></tr></table>

# 4.2.2 Cost-effective Inference on RTX 4090

The NVIDIA GeForce RTX 4090 is a highly cost-effective GPU with 24G memory. However, through in-depth memory profiling and analysis, we identified memory insufficiency as the primary bottleneck to serve our model on it. To address this challenge, we developed a highly memory-eficient inference architecture and performs systematically optimizations. As a result, we successfully deployed and ran our 4.5B-parameter model on a single RTX 4090 GPU, and also support our largest 24B model on an $\bar { 8 \times }$ RTX 4090 GPUs. In the following section, we briefly introduce the key optimization techniques.

Memory Optimization To address the memory constraints of the RTX 4090, we used a variety of techniques to do systematically optimization:

•Quantization: We adopt the same quantization strategy (WA8A SmoothQuant) as for streaming video generation.   
• KV-offload: KV-offload is a technique that stores the KV cache in CPU memory by default and dynamically re-load it back to the GPU as needed. This approach significantly reduces peak GPU memory usage and is widely adopted in long-sequence processing for large language models (LLMs). In MAGI-1, we also adopt this technique to effectively address memory constraints.   
Hybrid Parallism and Communication Optimization: The above two optimizations only enable 4.5B model deployment on a single RTX 4090 GPU. However, the largest 24B model further requires multi-GPU parallelism. Unlike the streaming setting where we primarily adopt a Ulysses-based context-parallelism (CP) approach, deployment

on RTX 4090 employs a hybrid strategy combining pipeline-parallelism (PP) and context-parallelism.

Specifically, pipeline-parallelism is used to partition model weights, while contextparallelism is used to partition activations. However, since the RTX 4090 utilizes PCIe for inter-GPU communication, both PP and CP suffer from communication-induced bubbles that degrade compute utilization, as measured by Model FLOPs Utilization (MFU). For PP, we mitigate this by interleaving tasks to overlap GPU idle. For contextparallelism, we initially adopted the Ulysses approach, but found that communication could not be fully overlapped with computation under PCIe constraints.

Therefore, we propose an enhancement to Ulysses called Context Shuffle Overlap (CSO)(Details in Sec. A.3), which scatters each chunk evenly across all GPUs, enabling finer-grained overlap between computation and communication than plain Ulysses. This strategy significantly improves MFU under the limited interconnect bandwidth of the RTX 4090.

With the above optimizations, we constrained peak memory usage to 19.07GB for the 4.5B model on a single RTX 4090 GPU, and 19.29 GB for the 24B model on $8 \times$ RTX 4090 GPUs. For the 24B model, the maximum MFU reached $6 6 \%$ .

# 5 Evaluation

Evaluation methods for video generation models in the research community are typically categorized into two complementary types: the first focuses on the perceptual quality of the generated videos, while the second evaluates the model's ability to faithfully capture underlying physics, which is often regarded as essential for modeling a realistic world. In MAGI-1, we adopt both evaluation types to obtain a comprehensive understanding of the model's strengths and limitations.

For perceptual quality evaluation, the inherently subjective nature of human preference, combined with the high-dimensional and diverse characteristics of video content (e.g., motion continuity, aesthetic, and identity consistency), makes it challenging to rely solely on objective metrics. As a result, the community typically employs a hybrid evaluation protocol that integrates human subjective assessments with standardized automated metrics, ensuring a more robust and comprehensive evaluation.

There is currently no universally accepted human evaluation protocol or human evaluation platform within the community for perceptual quality evaluation. To address this, we design our own in-house evaluation benchmark based on a comprehensive review of existing human evaluation methodologies, combined with our understanding of both evaluation criteria and model capabilities. Human experts serve as evaluator in this system, comparing our model against other competitors under strict double-blind conditions, and providing assessments across multiple perceptual dimensions. For objective evaluation, we adopt VBench (Huang et al., 2024), which is currently the most widely used benchmark in the community. VBench consists of two evaluation tracks: text-to-video (T2V) and image-tovideo (2V). We primarily focus on the 2V track, as it more closely reflects real-world usage patterns: users typically generate videos from images rather than from text. For the same reason, we also allocate a larger proportion of I2V tasks during the training of MAGI-1, aiming to better align the model's capabilities with practical deployment scenarios.

Physics-IQ (Motamed et al., 2025) is one of the most representative benchmarks for evaluating a model's ability to capture physical dynamics in video. It presents a short video clip depicting real-world physical motion and asks the model to predict future frames. The predictions are then compared against ground-truth sequences to assess the model's understanding of physical rules.

The evaluation framework and the corresponding benchmark metrics are summarized in Tab. 7. The following sections present our evaluations in detail, and if not specified, we evaluate our 24B model by default.

Table 7: Evaluation Benchmark Overview   

<table><tr><td rowspan=1 colspan=1>Evaluation Category</td><td rowspan=1 colspan=1>Benchmark</td><td rowspan=1 colspan=1>Metrics</td></tr><tr><td rowspan=2 colspan=1>PerceptualEvaluation</td><td rowspan=1 colspan=1>In-house Human Evaluation</td><td rowspan=1 colspan=1>OverallMotion QualityInstruction FollowingVisual Quality</td></tr><tr><td rowspan=1 colspan=1>VBench-I2V</td><td rowspan=1 colspan=1>Automated Quality Metrics</td></tr><tr><td rowspan=1 colspan=1>PhysicalEvaluation</td><td rowspan=1 colspan=1>Physics-IQ-Benchmark</td><td rowspan=1 colspan=1>Physics-IQ-Score</td></tr></table>

# 5.1 Perceptual Evaluation

# 5.1.1 In-house Human Evaluation Benchmark

Our in-house evaluation benchmark is primarily designed for I2V task, and integrates three complementary components to ensure comprehensive and unbiased assessment. First, we design a hierarchical metric system that prioritizes completeness over simplicity, while enforcing orthogonality among metrics to enable fine-grained evaluation across multiple quality dimensions without redundancy. Second, we construct a benchmark dataset of 100 diverse image-prompt pairs through systematic selection. These pairs span a broad spectrum of scenarios, from simple object motions to complex human activities, and each curated to probe specific aspects of video generation capability. Third, we implement a double-blind comparison protocol with standardized output normalization, ensuring that each model operates under fair conditions for a meaningful comparison.

Evaluation Metrics. To ensure a comprehensive and reliable evaluation while avoiding unnecessary complexity, we adhere to three guiding principles in our metric design: comprehensiveness first, simplicity second, and orthogonality third. Unlike T2V, where both visual content and motion are generated from scratch, I2V starts with a fixed visual input provided by the user's uploaded image, while the subsequent dynamics are guided by the input text condition. This distinction shifts the evaluation focus toward assessing the motion and temporal quality of generated video while ensuring faithful preservation of the original visual elements.

Through preliminary analysis, we identified several common failure modes in I2V generation, including distortion, clipping, and temporal jittering. These typical issues guided the design of our evaluation framework, which emphasizes motion quality, temporal coherence, and the trade-off between source image fidelity and natural animation." Therefore, our evaluation framework organizes metrics into four primarily dimensions: Overall, Motion Quality, Instruction Following, and Visual Quality. Each dimension is further broken down into specific sub-metrics designed to capture particular aspects of video generation quality as shown in Tab. 8.

Dataset Construction. We construct a benchmark dataset consisting of 100 high-quality image-prompt pairs, each carefully selected to challenge different aspects of I2V generation. To ensure diversity and representativeness, we source data from four sources: 1) usersubmitted inputs from existing video generation platforms, 2) synthetic images generated by FLUX (Labs, 2024), 3) authentic photographs from public repositories, and 4) professional cinematographic materials. Each sample is annotated with specific evaluation targets defined by our metric framework, enabling broad coverage of assessment dimensions while avoiding redundancy.

The dataset construction process follows a systematic multi-stage pipeline. We first establish a set of selection criteria focused on key challenges in I2V generation, including complex object deformation, multi-object interaction, dynamic camera motion, and lighting transitions. Based on these criteria, experts nominate candidate samples, which are then finalized through a collaborative voting procedure. This curated process ensures the resultin benchmark presents a diverse ye focused set o evaluation cases for gorously ttin I2V models.

Table 8: Hierarchical Evaluation Framework   

<table><tr><td rowspan=1 colspan=1>MainMetric</td><td rowspan=1 colspan=1>SubMetric</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>Overall</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>General preference</td></tr><tr><td rowspan=1 colspan=1>MotionQuality</td><td rowspan=1 colspan=1>Motion SpeedMotion AmplitudeMotion SmoothnessMovement Direction</td><td rowspan=1 colspan=1>Appropriate timing of movementsNatural range of movementContinuous movement without jitterLogical and consistent direction</td></tr><tr><td rowspan=1 colspan=1>InstructionFollowing</td><td rowspan=1 colspan=1>Subject AdherenceEnvironment AdherenceCamera Adherence</td><td rowspan=1 colspan=1>Following behavioral instructionsMeeting contextual requirementsFollowing camera movement requests</td></tr><tr><td rowspan=1 colspan=1>VisualQuality</td><td rowspan=1 colspan=1>Subject FeaturesScene FeaturesLighting ChangesTexture Changes</td><td rowspan=1 colspan=1>Consistency of main subjectConsistency of environmentQuality of lighting transitionsConsistency of surface appearances</td></tr></table>

Results and Analysis. Our evaluation methodology employs a paired comparison approach designed to directly measure relative model performance. Specifically, for each test case, we generate two videos (one from our model and one from a comparative model) using identical prompts and input images. Expert evaluators with strong aesthetic training then indicate their preference between each pair (Win/Tie/Lose) across multiple evaluation dimensions without knowledge of which model produced which video.

MAGI-1's autoregressive design enables generation of arbitrary-length videos. For fair comparison, we adapt our generation length to match each comparison model: for example, 5 seconds for Kling and 6 seconds for Hailuo. To avoid potential manipulation of visual quality, we maintain each model's native output without post-processing like resolution normalization. In addition, all models are evaluated using raw user inputs without any manual refinement from our side, relying solely on their built-in prompt enhancement (PE) mechanisms.

The evaluation results shown in Fig. 16 demonstrate MAGI-1's strong competitive position in the field. In terms of overall performance, our model shows advantages over the open-source model Wan-2.1 (Wang et al., 2025a), performs slightly behind the commercial model Kling1.6 (HD) (Kuaishou, 2024), but achieves clearly better results compared to both Hailuo(i2v01) (MiniMax, 2024) and HunyuanVideo (Kong et al., 2024). Looking at specific capabilities, MAGI-1 excels particularly in instruction following and motion quality metrics, consistently receiving high scores across comparisons. However, in terms of visual quality, there remains room for improvement compared to top models.

# 5.1.2 VBench

VBench (Huang et al., 2024) is currently the most widely adopted benchmark in the community for automated and objective evaluation of video generation models. While its evaluation framework is still evolving and not without limitations, VBench remains a critical tool for model comparison due to its fully automated and reproducible assessment process, especially when contrasted with in-house human evaluations, which are often subjective and lack transparency.

VBench provides two primary evaluation tracks: text-to-video (T2V) and image-to-video (I2V). Given that I2V more closely reflects real-world usage patterns, where users typically input a static image to generate videos in existing product, and in line with our goal of aligning evaluation with practical application scenarios, we focus our evaluation on the I2V track in VBench.

![](images/23.jpg)  
Figure 16: Comparative evaluation of our model against leading open-source and proprietary video generation models across multiple metrics. Each bar is divided into three sections: red, gray, and blue, representing Win-Tie-Loss percentages for each comparison. Blue sections indicate where users preferred the competitor model, gray sections represent ties, and red sections show where users preferred our model. The evaluation includes both API-based assessments like Kling1.6 (HD) (Kuaishou, 2024) and Hailuo (i2v01) (MiniMax, 2024) and locally deployed models like Wan-2.1 (Wang et al., 2025a) and HunyuanVideo (Kong et al., 2024)), providing a comprehensive comparison across various implementation environments.

We evaluate the generation quality of MAGI-1 under two different configurations: MAGI-1 ( $1 \times$ decoder) and MAGI-1 $2 \bar { \times }$ decoder). The only difference between them lies in the VAE decoder: MAGI-1 $2 \times$ decder) employs an enhanced decoder capable of $2 \times$ upsampling, while the core autoregressive denoising model remains identical across both versions. For evaluation, both models generate 4-second videos at 24 FPS with a 16:9 aspect ratio.

The results are presented in Tab. 9. As shown, both of our models achieve outstanding performance, with MAGI-1 $2 \times$ decoder) reaching a top overall score of 89.28, ranking first among all models. Notably, the MAGI-1 models demonstrate a significant advantage in ednam Degree compard to other approace, while simultaneously maintainng high visual quality, including strong performance in aesthetic quality and motion smoothness. This effectively addresses a common trade-off in other methods, where increasing motion amplitude often downgrade image quality. We attribute this strength to the autoregressive denoising architecture, which provides a stronger modeling capability for complex motion dynamics.

# 5.2 Physical Evaluation

Video generation models are increasingly recognized as a foundation toward building the world model, and the ability to accurately capture real-world physical dynamics has become a central focus within the research community. In contrast to perceptual evaluation, which inevitably involves subjective human preferences, physics-based evaluation aims to assess a model's ability to understand and simulate objective physical principles.

Currently, there are only a few established benchmarks (Bansal et al., 2024; Meng et al., 2024; Dash et al., 2011; Yi et al., 2019; Kang et al., 2024) in this area, and Physics-IQ (Motamed et al., 2025) stands out as the most comprehensive and state-of-the-art benchmark. Therefore, we adopt Physics-IQ to evaluate the physical understanding and reasoning capabilities of MAGI-1.

Table 9: Quantitative evaluation results on VBench-I2V benchmark. MAGI-1 ( $1 \times$ decoder) denotes our baseline model $( 1 2 8 0 \times 7 2 0$ resolution), while MAGI-1 $2 \times$ decoder) represents the enhanced variant with $2 \mathrm { x }$ VAE upsampling $( 2 5 6 0 \times 1 4 4 0$ resolution). Comparative data for other models are sourced from the top tier at latest Vbench leaderboard. Bold and underlined values indicate the highest and second-highest scores respectively across all metrics.   

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Metric(VBenchI2V)</td><td rowspan=1 colspan=1>MAGI-1(2× decoder)</td><td rowspan=1 colspan=1>MAGI-1(1× decoder)</td><td rowspan=1 colspan=1>VisualPi</td><td rowspan=1 colspan=1>StepFun(TI2V)</td></tr><tr><td rowspan=8 colspan=1>QualityMetrics</td><td rowspan=8 colspan=3>I2V-CameraI2V-Subject12V-BackgroundSubject Cons.Motion Smooth.Imaging QualityDynamic DegreeBackground Cons.Aesthetic Quality</td><td rowspan=1 colspan=1>50.85</td><td rowspan=1 colspan=1>50.77</td><td rowspan=1 colspan=1>51.20</td><td rowspan=8 colspan=1>49.2397.8698.6396.0299.2470.4448.7897.0662.29</td></tr><tr><td rowspan=1 colspan=1>I2V-Subject</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>98.39</td><td rowspan=1 colspan=1>98.36</td><td rowspan=1 colspan=1>98.67</td></tr><tr><td rowspan=6 colspan=1>99.0093.9698.6869.7168.2196.7464.74</td><td rowspan=1 colspan=1>98.98</td><td rowspan=1 colspan=1>98.87</td></tr><tr><td rowspan=1 colspan=1>94.28</td><td rowspan=1 colspan=1>96.87</td></tr><tr><td rowspan=1 colspan=1>98.83</td><td rowspan=4 colspan=1>99.1872.8649.9397.5061.91</td></tr><tr><td rowspan=1 colspan=1>69.68</td></tr><tr><td rowspan=1 colspan=1>63.41</td></tr><tr><td rowspan=1 colspan=1>96.9061.89</td></tr><tr><td rowspan=2 colspan=1>Agg.Scores</td><td rowspan=2 colspan=3>Quality ScoreI2V ScoreTotal Score</td><td rowspan=2 colspan=1>82.4496.1289.28</td><td rowspan=1 colspan=1>81.6796.08</td><td rowspan=1 colspan=1>81.9596.21</td><td rowspan=2 colspan=1>81.2295.5088.36</td></tr><tr><td rowspan=1 colspan=1>88.88</td><td rowspan=1 colspan=1>89.08</td></tr></table>

The Physics-IQ evaluation protocol uses 8-second real-world videos that depict objective physical phenomena. The first 3 seconds of each video are provided as conditional input to the model, which is then required to predict the remaining 5 seconds. The accuracy of the model's physical modeling capability is measured by comparing the predicted videos with the ground truth.

Since most existing video generation models do not natively support video-conditioned continuation, they typically approximate this task using image-to-video (I2V) generation, conditioning only on the last frame of the input video. To provide a comprehensive comparison, we report results for both two settings.

The results are presented in Tab. 10. When conditioned on video inputs, MAGI-1 outperforms all competing models by a substantial margin, reaches the score of 56.02. The previous state-of-the-art model VideoPoet (Kondratyuk et al., 2023), which also supports video-to-video (V2V) prediction, is outperformed by approximately 27 points. Even when using only image condition, MAGI-1 still achieves the highest score among all models, reaching 30.23, despite a noticeable drop compared to its video-conditioned version.

These results clearly demonstrate the strong capability of MAGI-1 in understanding and modeling real-world physical principles. We attribute this advantage to its autoregressive nature: modeling physical processes demands a focus on causality rather than mere correlation, and autoregressive models inherently promote causal reasoning. In contrast, bidirectional denoising models lack the algorithmic foundations necessary to effectively capture causality, which leads to inferior performance in such tasks. While VideoPoet is also an autoregressive model, its primary design objective is integration with language models, which limits its efficiency in modeling the video modality; In contrast, MAGi-1 is purposebuilt for video generation, combining the strengths of autoregressive and denoising-based modeling. This targeted design enables it to achieve significantly superior performance.

Nevertheless, our model is not without limitations. Fig. 17 presents several representative results, revealing both its strengths and weaknesses. While MAGI-1 effectively captures primary dynamics—such as projectile motion, rotational behavior, and material deformation, it struggles with complex secondary effects, including precise collision responses, material-specific reactions, and post-deformation behavior. Notably, even when the predicted outcome deviates from the ground truth, the model often generates physically

<table><tr><td rowspan=1 colspan=2>Model</td><td rowspan=1 colspan=1>Phys.IQ Score</td><td rowspan=1 colspan=1>SpatialIoU ↑</td><td rowspan=1 colspan=1>SpatioTemporal↑</td><td rowspan=1 colspan=1>WeightedSpatial IoU ↑</td><td rowspan=1 colspan=1>MSE↓</td></tr><tr><td rowspan=3 colspan=2>MagI-1 (V2V)VideoPoet (V2V) (Kondratyuk et al., 2023)Lumiere (V2V) (Bar-Tal et al., 2024)</td><td rowspan=3 colspan=1>56.0229.5023.00</td><td rowspan=3 colspan=1>0.3670.2040.170</td><td rowspan=1 colspan=1>0.270</td><td rowspan=1 colspan=1>0.304</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=1>0.164</td><td rowspan=2 colspan=1>0.1370.093</td><td rowspan=1 colspan=1>0.010</td></tr><tr><td rowspan=1 colspan=1>0.155</td><td rowspan=1 colspan=1>0.013</td></tr><tr><td rowspan=9 colspan=2>MAGI-1 (I2V)Kling1.6 (I2V) (Kuaishou, 2024)VideoPoet (I2V) (Kondratyuk et al., 2023)Gen 3 (I2V) (Runway, 2024)Wan2.1 (I2V) (Wang et al., 2025a)Lumiere (I2V) (Bar-Tal et al., 2024)SVD (I2V) (Blattmann et al., 2023)Pika 1.0 (I2V) (PikaLabs, 2024)Sora (I2V) (OpenAI, 2024)</td><td rowspan=2 colspan=1>30.2323.64</td><td rowspan=1 colspan=1>0.203</td><td rowspan=1 colspan=1>0.151</td><td rowspan=1 colspan=1>0.154</td><td rowspan=2 colspan=1>0.0120.025</td></tr><tr><td rowspan=1 colspan=1>0.197</td><td rowspan=1 colspan=1>0.086</td><td rowspan=1 colspan=1>0.144</td></tr><tr><td rowspan=1 colspan=1>20.30</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.126</td><td rowspan=1 colspan=1>0.087</td><td rowspan=1 colspan=1>0.012</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>22.80</td><td rowspan=1 colspan=1>0.201</td><td rowspan=1 colspan=1>0.115</td><td rowspan=1 colspan=1>0.116</td><td rowspan=1 colspan=1>0.015</td></tr><tr><td rowspan=1 colspan=1>20.89</td><td rowspan=1 colspan=1>0.153</td><td rowspan=1 colspan=1>0.100</td><td rowspan=2 colspan=1>0.1120.061</td><td rowspan=1 colspan=1>0.023</td></tr><tr><td rowspan=1 colspan=1>0.113</td><td rowspan=1 colspan=1>0.173</td><td rowspan=1 colspan=1>0.016</td></tr><tr><td rowspan=1 colspan=1>0.132</td><td rowspan=1 colspan=1>0.076</td><td rowspan=1 colspan=1>0.073</td><td rowspan=1 colspan=1>0.021</td></tr><tr><td rowspan=1 colspan=1>13.00</td><td rowspan=1 colspan=1>0.140</td><td rowspan=1 colspan=1>0.041</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.014</td></tr><tr><td rowspan=1 colspan=1>10.00</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.047</td><td rowspan=1 colspan=1>0.063</td><td rowspan=1 colspan=1>0.030</td></tr><tr><td rowspan=1 colspan=2>GroundTruth</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.678</td><td rowspan=1 colspan=1>0.535</td><td rowspan=1 colspan=1>0.577</td><td rowspan=1 colspan=1>0.002</td></tr></table>

Table 10: Quantitative comparison of video generation models evaluated on the PhysicsIQ-Benchmark. Models are categorized by input modality: image-to-video (I2V) and video-to-video (V2V). Results were obtained through direct evaluation of model APIs, local deployment of open-source implementations, and as reported in Motamed et al. (2025). In the V2V task, models observe the first 3 seconds of an 8-second ground truth video and predict the remaining 5 seconds, while in the I2V task, models take only a single frame at the 3-second mark and predict the subsequent 5 seconds. Magi(V2V) utilizes the full 24 FPS video input (96 frames).

plausible alternatives. For example, in the second case (Fig. 17(b)), although the model fails to simulate the ignition of a match and the popping of a balloon, it instead produces a coherent sequence in which the rod rotates, contacts the object, and realistically bends upon impact. These results suggest that MAGI-1 has acquired a non-trivial physical intuition, capable of generating alternative yet physically consistent scenarios.

The influence of historical context length The benefit of utilizing historical context for more accurate predictions has already been demonstrated in the comparison between imageconditioned and video-conditioned MAGI-1 models. To more systematically evaluate the impact of historical information in physical modeling, we varied the length of accessible history by adjusting the KV range of MAGI-1 during inference. Fig. 18 presents the results. Overall, we observe that increasing the amount of historical context generally leads to better performance. However, the most significant gain occurs at KV range $= \dot { 2 }$ ,meaning that short-term history is often sufficient to support accurate predictions.

# 6 Related Works

This section reviews major developments in text-to-video generation, categorized by proprietary systems, open-source efforts, and recent trends in autoregressive and causal modeling. We highlight unresolved challenges in scalability, causality, and streaming compatibility—challenges that MAGI-1 is designed to address.

Proprietary Systems. Recent proprietary models have significantly advanced generation length, resolution, and semantic fidelity. OpenAI's Sora (OpenAI, 2024) introduced long-form, high-resolution generation with strong prompt consistency. Kuaishou's Kling (Kuaishou, 2024) and Runway's Gen-3 (Runway, 2024) emphasized temporal fidelity and fine-grained stylistic control, respectively. Luma AI's DreamMachine (LumaLabs, 2024) improved motion continuity and stylistic adherence. Pika Labs' Pika 1.5 (PikaLabs, 2024) enabled interactive control over visual attributes, while Meta's MovieGen (Polyak et al., 2024) offered transparency into foundational model training. Most recently, Google's Veo 2 (DeepMind, 2024) advanced physical realism and human motion modeling. Despite

0.0s 2.5s 3.0s 3.3s 3.7s 4.0s 4.3s 4.7s 5.0s 7.9s 20

(a) A light beige coffee table with a small yellow rubber ducky on it. A mustard yellow couch is in the background. There is a black pipe on one end of the table and a brown tennis ball rolls out of it towards the rubber ducky. Static shot with no camera movement.

0.0s 2.5s 3.0s 3.3s 3.7s 4.0s 4.3s 4.7s 5.0s 7.9s 20

(b) A black balloon is sitting on a wooden table next to a small rotating platform with a lit matchstick taped to it. The match rotates clockwise and touches the ballon. Static shot with no camera movement.

![](images/24.jpg)

(c) Two black and blue gripping tools are pulling a piece of green paper from its two corners, causing it to tear. Static shot with no camera movement.

Figure 17: Case study results from the Physics-IQ Benchmark illustrate three distinct physical scenarios over time. Each scenario compares the ground truth (top row) with our model's predictions (bottom row), conditioned on the first 3 seconds and forecasting the next 5 seconds. The results highlight the model's ability to capture core physical interactions, as well as its limitations with complex material-specific effects: (a) The model correctly predicts the initial projectile motion but erroneously shows the ball deflecting off the duck instead of stopping upon impact. (b) Rotational dynamics are accurately captured, but the model fails to predict the match igniting and popping the balloon, instead showing the object being pushed back. (c) The model predicts the card tearing but struggles to model the motion of the torn pieces afterward.

![](images/25.jpg)  
Figure 18: Physical IQ scores as a function of historical context. This visualization shows how performance changes with varying amounts of historical information, represented by the KV Range Value.

these innovations, most systems are closed-source and opaque in architecture, limiting reproducibility and extensibility.

Open-Source Ecosystem. The open-source community pioneered latent diffusion through Stable Diffusion (Esser et al., 2021), which integrated a variational autoencoder (Kingma, 2013) for latent representation, a CLIP-based text encoder (Radford et al., 2021), and a U-Net denoiser (Ronneberger et al., 2015). Temporal extensions such as VDM (Ho et al., 2022), AnimateDiff (Guo et al., 2024), and SVD (Blattmann et al., 2023) adapted the architecture for frame coherence. Transformer-based backbones like DiT (Peebles & Xie, 2023), PixArt- $\alpha$ (Chen et al., 2023), and Latte (Ma et al., 2024) demonstrated scalability and inspired early video adaptations. Recent open implementations—including Open-Sora (Zheng et al., 2024), Open-Sora-Plan (Lin et al., 2024), CogVideoX (Yang et al., 2025), Mochi 1 (GenmoTeam, 2024), HunyuanVideo (Kong et al., 2024), StepVideo (Ma et al., 2025), LTX-Video (HaCohen et al., 2024), and Wan (Wang et al., 2025a)—introduced modular advances in chunking, compression, and streaming. However, these systems largely retain bidirectional denoising and globally conditioned inference, limiting applicability to real-time or causal settings.

Autoregressive and Causal Modeling. An emerging trend is the integration of autoregressive modeling and causal constraints. Diffusion Forcing (Chen et al., 2024a) introduces independent per-token noise schedules that allow a causal model to denoise future tokens while keeping past tokens minimally perturbed, effectively unifying next-token prediction with full-sequence diffusion. FVDM (Liu et al., 2024) employed timestep vectorization for precise noise control. CausVid (Yin et al., 2024) combined causal inference with distillation for streaming scenarios. While promising, these models remain limited in scale, often lack chunk-wise abstraction, and do not unify video continuation with I2V /T2V generation.

MAGI-1: Scalable Autoregressive Diffusion. To our knowledge, MAGI-1 is the first large-scale, chunk-wise autoregressive diffusion model trained from scratch that unifies high-fidelity text-to-video, image-to-video, and video continuation tasks under strict causal constraints. It supports real-time streaming and long-horizon synthesis via efficient chunkwise denoising, shortcut distillation, and KV-cached inference. By explicitly addressing scalability, causality, and streaming compatibility, MAGI-1 establishes a new foundation for unified and controllable video generation.

# 7 Conclusion

MAGI-1 introduces a scalable chunk-wise autoregressive diffusion framework for highfidelity video synthesis. By progressively denoising fixed-length segments under strict causal constraints, it enables real-time, streaming-compatible generation with fixed computational overhead regardless of video length. The architecture builds upon a Transformer backbone enhanced with block-causal and parallel attention modules, and is supported by a distributed attention mechanism and a highly efficient training strategy for handling ultra-long contexts.

A key contribution lies in its unified design: MAGI-1 supports text-to-video, image-to-video, and video continuation tasks without requiring task-specific modifications, all under a shared training objective. Through chunk-wise text conditioning, it further achieves finegrained semantic control across long-form video generation. A shortcut distillation strategy significantly reduces the number of diffusion steps required for inference, improving efficiency while maintaining temporal consistency and sample quality.

Empirical results on VBench-I2V and Physics-IQ benchmarks demonstrate that MAGI-1 outperforms existing large-scale video diffusion models in prompt adherence, physical plausibility, and temporal coherence. Taken together, these contributions establish MAGI-1 as a robust and extensible foundation for autoregressive video synthesis—offering both state-of-the-art performance and a fertile ground for future advancements in modularity, controllability, and multi-modal reasoning.

# 8 Limitation and Future Work

While MAGI-1 demonstrates strong generation quality and low-latency inference via chunkwise autoregressive denoising, its current architecture remains tightly coupled. Specifically, a single large decoder-style Transformer is tasked with both (1) high-level temporal context fusion—integrating static conditioning signals with progressively noisier visual inputs—and (2) low-level denoising, which requires accurate reconstruction of fine-grained visual details. This conflation of heterogeneous objectives introduces several technical limitations:

•Inference latency bottleneck: The same large model is repeatedly invoked across all denoising steps, even when only minor refinements are required. This leads to inefficient utilization of compute, especially in streaming settings where low-latency frame delivery is critical.

•Optimization conflict: Jointly optimizing global semantic planning and pixel-level restoration within a single model exacerbates objective interference, often leading to suboptimal scaling behavior.

Limited controllability: The monolithic architecture constrains the insertion of auxiliary control signals—such as confidence-based guidance modulation, or dynamic temporal constraints—due to entangled latent pathways and overlapping functional scopes.

Thus, a decoupled design that structurally separates high-level semantic reasoning from low-level visual synthesis is worth exploring. Looking ahead, as video generation evolves from producing isolated clips to constructing long-form content with coherent narratives, we anticipate a convergence between video generation and understanding. In this closed-loop setting, the quality of generated content will increasingly depend on the model's capacity to understand video content, making understanding the key bottleneck. Although we are il    e toward closing the loop between video understanding and generation.