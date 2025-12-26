# KV-Edit: Training-Free Image Editing for Precise Background Preservation

Tianrui Zhu1\*, Shiyi Zhang1, Jiawei Shao2, Yansong Tang1\* 1Shenzhen International Graduate School, Tsinghua University 2Institute of Artificial Intelligence (TeleAI), China Telecom xilluill070513@gmail.com,sy-zhang23@mails.tsinghua.edu.cn shaojw2@chinatelecom.cn,tang.yansong@sz.tsinghua.edu.cn https://xilluill.github.io/projectpages/KV-Edit/

![](images/1.jpg)  
.

# Abstract

Background consistency remains a significant challenge in image editing tasks. Despite extensive developments, existing works still face a trade-off between maintaining similarity to the original image and generating content that aligns with the target. Here, we propose KV-Edit, a training-free approach that uses KV cache in DiTs to maintain background consistency, where background tokens are preserved rather than regenerated, eliminating the need for complex mechanisms or expensive training, ultimately generating new content that seamlessly integrates with the background within user-provided regions. We further explore the memory consumption of the KV cache during editing and optimize the space complexity to $O ( 1 )$ using an inversion-free method. Our approach is compatible with any DiT-based generative model without additional training. Experiments demonstrate that KV-Edit significantly outperforms existing approaches in terms of both background and image quality, even surpassing training-based methods.

# 1. Introduction

Recent advances in text-to-image (T2I) generation have witnessed a significant shift from UNet [43] to DiT [39] architectures, and from diffusion models (DMs) [11, 48, 52] to flow models (FMs) [1, 23, 64]. Flow-based models, such as Flux [1], construct a straight probability flow from noise to image, enabling faster generation with fewer sampling steps and reduced training resources. DiTs [39], with their pure attention architecture, have demonstrated superior generation quality and enhanced scalability compared to UNetbased models. These T2I models [1, 14, 42] can also facilitate image editing, where target images are generated based on source images and modified text prompts. In the field of image editing, early works [12, 16, 36, 50] proposed the inversion-denoising paradigm to generate edited images, but they struggle to maintain background consistency during editing. One popular approach is attention modification, such as HeadRouter [57] modifying attention maps and PnP [50] injecting original features during the denoising process, aiming to increase similarity with the source image. However, there remains a significant gap between improved similarity and perfect consistency, as it is challenging to control networks' behavior as intended. Another common approach is designing new samplers [37, 38] to reduce errors during inversion. Nevertheless, errors can only be reduced but not completely eliminated and both training-free approaches above still require extensive hyperparameter tuning for different cases. Meanwhile, exciting training-based inpainting methods [26, 65] can maintain background consistency but suffer from expensive training costs and potential degradation of quality. To overcome all the above limitations, we propose a new training-free method that preserves background consistency during editing. Instead of relying on regular attention modification or new inversion samplers for similar results, we implement KV cache in DiTs [39] to preserve the keyvalue pairs of background tokens during inversion and selectively reconstruct only the editing region. Our approach first employs a mask to decouple attention between background and foreground regions and then inverts the image into noise space while caching KV values of background tokens at each timestep and attention layer. During the subsequent denoising process, only foreground tokens are processed, while their keys and values are concatenated with the cached background information. Effectively, we guide the generative model to maintain new content continuity with the background and keep the background content identical to the input. We call this approach KV-Edit. To further enhance the practical utility of our approach, we conduct an analysis of the removal scenario. This challenge arises from the residual information in surrounding tokens and the object itself which sometimes conflict with the editing instruction. To address this issue, we introduce mask-guided inversion and reinitialization strategies as two enhancement techniques for inversion and denoising separately. These methods further disrupt the information stored in surrounding tokens and self tokens respectively, enabling better alignment with the text prompt. In addition, we apply KV-Edit to the inversion-free method [23, 56], which no longer caches key-value pairs for all timesteps, but uses KV immediately after one step, significantly reducing the memory consumption of the KV cache. In summary, our key contributions include: 1) A new training-free editing method that implements KV cache in DiTs, ensuring complete background consistency during editing with minimal hyperparameter tuning. 2) Maskguided inversion and reinitialization strategies that extend the method's applicability across various editing tasks, offering flexible choices for different user needs. 3) Using the inversion-free method to optimize the memory overhead of our method and enhance its usefulness on PC. 4) Experimental validation demonstrating perfect background preservation while maintaining generation quality comparable to direct T2I synthesis.

# 2. Related Work

# 2.1. Text-guidanced Editing

Image editing approaches can be broadly categorized into training-based and training-free methods. Training-based methods [1, 7, 20, 22, 26], have demonstrated impressive editing capabilities through fine-tuning pre-trained generative models on text-image pairs, achieving controlled modifications. Training-free methods have emerged as a flexible alternative, with pioneering works [12, 16, 36, 50] establishing the two-stage inversion-denoising paradigm. Attention modification has become a prevalent technique in these methods [4, 9, 25, 49, 57], specially Add-it [49] broadcast features from inversion to denoising process to maintain source image similarity during editing. Some other work [21, 27, 37, 38, 53] focused on a better inversion sampler such as the RF-solver [53] designs a second-order sampler. The methods most similar to ours [3, 10, 29, 49] attempt to preserve background elements by blending source and target images at specific timesteps using masks. A common consensus is that accurate masks are crucial for better quality, where user-provided inputs [20, 26] and segmentation models [6, 18, 34, 35, 41, 58, 59] prove to be more effective choices compared to masks derived from attention layers in UNet [43]. However, the above methods frequently encounter failure cases and struggle to maintain perfect background consistency during editing, while trainingbased methods [1, 7, 20, 22, 26, 65] face the additional challenge of computational overhead.

![](images/2.jpg)  
Here, $\mathbf { x }$ and $\mathbf { z }$ denote intermediate results in inversion and denoising processes respectively. Starting from $\mathbf { x } _ { 0 }$ , we first perform inversion to obtain predicted noise $\mathbf { x } _ { N }$ while caching KV pairs. Then, we choose the input ${ \bf z } _ { N } ^ { f g }$ and generate edited foreground content $\mathbf { z } _ { 0 } ^ { f g }$ based on a new prompt. Finally, we concatenate it with the original background $\mathbf { x } _ { 0 } ^ { b g }$ to obtain the edited image with preserved background.

# 2.2. KV cache in Attention Models

KV cache is a widely-adopted optimization technique in Large Language Models (LLMs) [5, 8, 30, 54] to improve the efficiency of autoregressive generation. In causal attention, since keys and values remain unchanged during generation, recomputing them leads to redundant resource consumption. KV cache addresses this by storing these intermediate results, allowing the model to reuse key-value pairs from previous tokens during inference. This technique has been successfully implemented in both LLMs [5, 8, 30, 54] and Vision Language Models (VLMs) [2, 17, 24, 31, 60 62]. However, it has not been explored in image generation and editing tasks, primarily because image tokens are typically assumed to require bidirectional attention [13, 15].

# 3. Method

In this section, we first analyze the reasons why the inversion-denoising paradigm [12, 16] faces challenges in background preservation. Then, we introduce the proposed KV-Edit method, which achieves strict preservation of background regions during the editing process according to the mask. Finally, we present two optional enhancement techniques and an inversion-free version to improve the usability of our method across diverse scenarios.

# 3.1. Preliminaries

Deterministic diffusion models like DDIM [46] and flow matching [28] can be modeled using ODE [47] to describe the probability flow path from noise distribution to real distribution. The model learns to predict velocity vectors that transform Gaussian noise into meaningful images. During the denoising process, $\mathbf { x _ { 1 } }$ represents noise, $\mathbf { x _ { 0 } }$ is the final image, and $\mathbf { x _ { t } }$ represents intermediate results.

$$
d \mathbf { x } _ { t } = \left( f ( \mathbf { x } _ { t } , t ) - \frac { 1 } { 2 } g ^ { 2 } ( t ) \nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { x } _ { t } ) \right) d t , t \in [ 0 , 1 ] .
$$

where $\mathbf { s } _ { \theta } ( \mathbf { x } , t ) ~ = ~ \nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { x } _ { t } )$ predicted by networks. Both DDIM [46] and flow matching [28] can be viewed as special cases of this ODE function. By setting $f ( \mathbf { x } _ { t } , t ) =$ $\begin{array} { r } { \frac { { \bf x } _ { t } } { \overline { { x } } _ { t } } \frac { d \overline { { \alpha } } _ { t } } { d t } , g ^ { 2 } ( t ) = 2 \overline { { \alpha } } _ { t } \overline { { \beta } } _ { t } \frac { d } { d t } \left( \frac { \overline { { \beta } } _ { t } } { \overline { { \alpha } } _ { t } } \right) } \end{array}$ and sθ(x, t) = − θ(,t), we obtain the discretized form of DDIM:

$$
\mathbf { x } _ { t - 1 } = \bar { \alpha } _ { t - 1 } \left( \frac { \mathbf { x } _ { t } - \bar { \beta } _ { t } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) } { \bar { \alpha } _ { t } } \right) + \bar { \beta } _ { t - 1 } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t )
$$

Both forward and reverse processes in ODE follow Eq. (1), describing a reversible path from Gaussian distribution to real distribution. During image editing, this ODE establishes a mapping between noise and real images, where noise can be viewed as an embedding of the image, carrying information about structure, semantics, and appearance. Recently, Rectified Flow [32, 33] constructs a straight path between noise distribution and real distribution, training a model to fit the velocity field $\mathbf { v } _ { \theta } ( \mathbf { x } , t )$ . This process can be simply described by the ODE:

$$
\begin{array} { r } { d { \mathbf { x } } _ { t } = { \mathbf { v } } _ { \theta } ( { \mathbf { x } } , t ) d t , t \in [ 0 , 1 ] . } \end{array}
$$

![](images/3.jpg)  
Figure 3. The reconstruction error in the inversionreconstruction process. Starting from the original image $\mathbf { x } _ { t _ { 0 } }$ , the inversion process proceeds to $\mathbf { x } _ { t _ { N } }$ . During inversion process, we use intermediate images $\mathbf { x } _ { t _ { i } }$ to reconstruct the original image and calculate the MSE between the reconstructed image $\mathbf { x } _ { t _ { 0 } } ^ { \prime }$ and the original image $\mathbf { x } _ { t _ { 0 } }$ .

Due to the reversible nature of ODEs, flow-based models can also be used for image editing through inversion and denoising in less timesteps than DDIM [46].

# 3.2. Rethinking the Inversion-Denoising Paradigm

The inversion-denoising paradigm views image editing as an inherent capability of generative models without additional training, capable of producing semantically different but visually similar images. However, empirical observations show that this paradigm only achieves similarity rather than perfect consistency in content, leaving a significant gap compared to users' expectations.This section will analyze the reasons for this issue into three factors.

Taking Rectified Flow [32, 33] as an example, based on Eq. (3), we can derive the discretized implementation of inversion and denoising. The model takes the original image $\mathbf { x } _ { t _ { 0 } }$ and Gaussian noise $\mathbf { x } _ { t _ { N } } \in \mathcal { N } ( 0 , I )$ as path endpoints. Given discrete timesteps $t = \{ t _ { N } , . . . , t _ { 0 } \}$ , the model predictions ${ \pmb v } _ { \theta } ( C , { \bf x } _ { t _ { i } } , t _ { i } ) , i \in \{ N , \cdot \cdot \cdot , 1 \}$ , where $\mathbf { x } _ { t _ { i } }$ and $\mathbf { z } _ { t _ { i } }$ denote intermediate states in inversion and denoising respectively, as described by the following equations:

$$
\begin{array} { r } { \mathbf { x } _ { t _ { i } } = \mathbf { x } _ { t _ { i - 1 } } + ( t _ { i } - t _ { i - 1 } ) \pmb { v } _ { \theta } ( C , \mathbf { x } _ { t _ { i } } , t _ { i } ) } \\ { \mathbf { z } _ { t _ { i - 1 } } = \mathbf { z } _ { t _ { i } } + ( t _ { i - 1 } - t _ { i } ) \pmb { v } _ { \theta } ( C , \mathbf { z } _ { t _ { i } } , t _ { i } ) } \end{array}
$$

Ideally, $\mathbf { z } _ { t _ { 0 } }$ should be identity with $\mathbf { x } _ { t _ { 0 } }$ when directly reconstructed from $\mathbf { x } _ { t _ { N } }$ . However, due to discretization and causality in the inversion process, we can only estimate using $v _ { \theta } ( C , \mathbf { X } _ { t _ { t - 1 } } , t _ { t - 1 } ) \approx v _ { \theta } ( C , \mathbf { X } _ { t _ { i } } , t _ { i } )$ , introducing cumulative errors. Fig. 3 shows that with a fixed number of timesteps $N$ , error accumulation increases as inversion timesteps approach $t _ { N }$ , preventing accurate reconstruction. dvide theo rnwe In addition, consistency is affected by condition. We can $\mathbf { z } _ { t _ { 0 } } ^ { f g }$ and regions we want to preserve $\mathbf { z } _ { t _ { 0 } } ^ { b g }$ , where "fg" and "bg" represent foreground and background respectively. Based on these definitions, the background denoising process is:

![](images/4.jpg)  
Figure 4. Analysis of factors affecting background changes. The four images on the right demonstrate how foreground content and condition changes influence the final results.

$$
{ \pmb v } _ { \theta } ( C , { \bf z } _ { t _ { i } } , t _ { i } ) = { \pmb v } _ { \theta } ( C , { \bf z } _ { t _ { i } } ^ { f g } , { \bf z } _ { t _ { i } } ^ { b g } , t _ { i } )
$$

$$
\mathbf { z } _ { t _ { i - 1 } } ^ { b g } = \mathbf { z } _ { t _ { i } } ^ { b g } + ( t _ { i - 1 } - t _ { i } ) \pmb { v } _ { \theta } ( C , \mathbf { z } _ { t _ { i } } ^ { f g } , \mathbf { z } _ { t _ { i } } ^ { b g } , t _ { i } )
$$

According to these formulas, when generating edited re$C$ ckgroind will be inf $\mathbf { z } _ { t _ { i } } ^ { f g }$ c ei by bon the new that background regions change when only modifying the prompt or foreground noise. In summary, uncontrollable background changes can be attributed to three factors: error accumulation, new conditions, and new foreground content. In practice, any single element will trigger all three effects simultaneously. Therefore, this paper will present an elegant solution to address all these issues simultaneously.

# 3.3. Attention Decoupling

Traditional inversion-denoising paradigms process background and foreground regions simultaneously during denoising, causing undesired background changes in response to foreground and condition modifications. Upon deeper analysis, we observe that in UNet [43] architectures, the extensive convolutional networks lead to the fusion of background and foreground information, making it impossible to separate them. However, in DiT [39], which primarily relies on attention blocks [51], allows us to use only foreground tokens as queries, generating foreground content separately and then combined with the background. Moreover, directly generating foreground tokens often results in discontinuous or incorrect content relative to the background. Therefore, we propose a new attention mechanism where queries contain only foreground information, while keys and values incorporate both foreground and

![](images/5.jpg)  
Figure 5. Demonstration of inversion-free KV-Edit. The right panel shows three comparative cases including a failure case, while the left panel illustrates inversion-free approach Significantly optimizes the space complexity to $O ( 1 )$ .

# Algorithm 1 Simplified KV cache during inversion

1: Input: $t _ { i }$ , image $\boldsymbol { x } _ { t _ { i } }$ , $M$ layer block $\{ l _ { j } \} _ { j = 1 } ^ { M }$ , fore  
ground region mask, KV cache $C$   
2: Output: Prediction vector $V _ { \theta t _ { i } }$ , $\mathrm { K V }$ cache $C$   
3: for $j = 0$ to $M$ do   
4: $\begin{array} { r l } & { Q , K , V = W _ { Q } ( x _ { t _ { i } } ) , W _ { K } ( x _ { t _ { i } } ) , W _ { V } ( x _ { t _ { i } } ) } \\ & { K _ { i j } ^ { b g } , V _ { i j } ^ { b g } = K [ 1 - m a s k > 0 ] , V [ 1 - m a s k > 0 ] } \\ & { C \gets \mathrm { A p p e n d } ( K _ { i j } ^ { b g } , V _ { i j } ^ { b g } ) } \\ & { x _ { t _ { i } } = x _ { t _ { i } } + \mathrm { A t t n } ( Q , K , V ) } \end{array}$   
5:   
6:   
7:   
8: end for   
9: $V _ { \theta t _ { i } } = \mathbf { M L P } ( x _ { t _ { i } } , t _ { i } )$   
10: Return $V _ { \theta t _ { i } }$ , $C$ background information. Excluding text tokens, the imagemodality self-attention computation can be expressed as:

$$
\operatorname { A t t } ( \mathbf { Q } ^ { f g } , ( \mathbf { K } ^ { f g } , \mathbf { K } ^ { b g } ) , ( \mathbf { V } ^ { f g } , \mathbf { V } ^ { b g } ) ) = \mathcal { S } ( \frac { \mathbf { Q } ^ { f g } \mathbf { K } ^ { T } } { \sqrt { d } } ) \mathbf { V }
$$

where $\mathbf { Q } ^ { f g }$ represents queries containing only foreground tokens, $( { \bf K } ^ { f g } , { \bf K } ^ { b g } )$ and $( { \bf V } ^ { f g } , { \bf V } ^ { b g } )$ denote the concatenation of background and foreground keys and values in their proper order (equivalent to the complete image's keys and values), and $s$ represents the softmax operation. Notably, compared to conventional attention computations, Eq. (8) only modifies the query component, which is equivalent to performing cropping at both input and output of the attention layer, ensuring seamless integration of the generated content with the background regions.

# Algorithm 2 Simplified KV cache during denosing

1: Input: $t _ { i }$ , foreground $z _ { t _ { i } } ^ { f g }$ , $M$ layer block $\{ l _ { j } \} _ { j = 1 } ^ { M }$ ,KV cache $C$

Output: Prediction vector $V _ { \theta t _ { i } } ^ { f g }$   
3: for $j = 0$ to $M$ do   
4: $Q ^ { f g } , K ^ { f g } , V ^ { f g } = W _ { Q } ( z _ { t _ { i } } ^ { f g } ) , W _ { K } ( z _ { t _ { i } } ^ { f g } ) , W _ { V } ( z _ { t _ { i } } ^ { f g } )$   
5: $K _ { i j } ^ { b g } , V _ { i j } ^ { b g } = C _ { K } [ i , j ] , C _ { V } [ i , j ]$   
6: $\bar { K , V } = \mathrm { C o n c a t } ( K _ { i j } ^ { b g } , K ^ { f g } ) , \mathrm { C o n c a t } ( V _ { i j } ^ { b g } , V ^ { f g } )$   
7: $z _ { t _ { i } } ^ { f g } = z _ { t _ { i } } ^ { f g } + \mathrm { A t t n } ( \stackrel { \cdot } { Q } ^ { f g } , K , V )$   
8: end for   
9: $V _ { \theta t _ { i } } ^ { f g } = \mathbf { M } \mathbf { L } \mathbf { P } ( z _ { t _ { i } } ^ { f g } , t _ { i } )$   
10: Return V f g θti

# 3.4. KV-Edit

Building upon Eq. (8), achieving background-preserving foreground editing requires providing appropriate key-value pairs for the background. Our core insight is that background tokens' keys and values reflect their deterministic path from image to noise. Therefore, we implement KV cache during the inversion process, as detailed in Algorithm 1. This approach records the keys and values at each timestep and block layer along the probability flow path, which are subsequently used during denoising as shown in Algorithm 2. We term this complete pipeline "KV-Edit" as shown in Fig. 2 where "KV" means KV cache. Unlike other attention injection methods [4, 49, 50], KVEdit only reuses KV for background tokens while regenerating foreground tokens, without requiring specification of particular attention layers or timesteps. Rather than using the source image as injected information, we treat the deterministic background as context and the foreground as content to continue generating, analogous to KV cache in LLMs. Since the background tokens are preserved rather than regenerated, KV-Edit ensures perfect background consistency, effectively circumventing the three influencing factors discussed in Sec. 3.2. Previous works [9, 12, 16] often fail in object removal tasks when using image captions as guidance, as the original object still aligns with the target prompt. Through our in-depth analysis, we reveal that this issue stems from the residual information of the original object, which persists both in its own tokens and propagates to surrounding tokens through attention mechanisms, ultimately leading the model to reconstruct the original content. To address the challenge in removing objects, we introduce two enhancement techniques. First, after inversion, we replace $\mathbf { z } _ { t _ { N } }$ with fused noise $\mathbf { z } _ { t _ { N } } ^ { \prime } = \mathrm { n o i s e } { \cdot } t _ { N } { + } \mathbf { z } _ { t _ { N } } { \cdot } ( 1 { - } t _ { N } )$ to disrupt the original content information. Second, we incorporate an attention mask during the inversion process, as illustrated in Fig. 2, to prevent foreground content from being incorporated into the KV values, further reducing the preservation of original content. These techniques serve as optional enhancements to improve editing capabilities and performances in different scenarios as shown in Fig. 1.

![](images/6.jpg)

# 3.5. Memory-Efficient Implementation

Inversion-based methods require storing key-value pairs for N timesteps, which can pose significant memory constraints when working with large-scale generative models (e.g., 12B parameters [1]) on personal computers. Fortunately, inspired by [23, 56], we explore an inversion-free approach. The method performs denoising immediately after each inversion step, computing the vector difference between the two results to derive a probability flow path in the $t _ { 0 }$ space. This approach allows immediate release of KV cache after use, reducing memory complexity from $O ( N )$ to $O ( 1 )$ . However, the inversion-free method may occasionally result in content retention artifacts as shown in Fig. 5 and FlowEdit [23]. Since our primary focus is investigating background preservation during editing, we leave more discussion about inversion-free in supplementary materials.

# 4. Experiments

# 4.1. Experimental Setup

Baselines. We compare our method against two categories of approaches: (1) Training-free methods including P2P [16], MasaCtrl [9] based on DDIM [46], and RFEdit [53], RF-Inversion [44] based on Rectified Flow [33]; (2) Training-based methods including BrushEdit [26] and FLUX-Fill [1], which are based on DDIM and Rectified Flow respectively. In total, we evaluate against six prevalent image editing and inpainting approaches. Datasets. We evaluate our method and baselines on nine tasks from PIE-Bench [21], which comprises 620 images with corresponding masks and text prompts. Following [26, 56], we exclude style transfer tasks from PIE-Bench [21] as our primary focus is background preservation in semantic editing tasks such as object addition, removal, and change.

Table 1. Comparison with previous methods on PIE-Bench. $\mathrm { V A E ^ { * } }$ denotes the inherent reconstruction error through direct VAE Bold and underlined values denote the best and second-best results respectively.   

<table><tr><td rowspan="2">Method</td><td colspan="2">Image Quality</td><td colspan="3">Masked Region Preservation</td><td colspan="2">Text Align</td></tr><tr><td>HPS×102 ↑</td><td>AS ↑</td><td>PSNR ↑</td><td>LPIPS×103 ↓</td><td>MSE×104 ↓</td><td>CLIP Sim ↑</td><td>IR×10 ↑</td></tr><tr><td>VAE*</td><td>24.93</td><td>6.37</td><td>37.65</td><td>7.93</td><td>3.86</td><td>19.69</td><td>-3.65</td></tr><tr><td>P2P [16]</td><td>25.40</td><td>6.27</td><td>17.86</td><td>208.43</td><td>219.22</td><td>22.24</td><td>0.017</td></tr><tr><td>MasaCtrl [9]</td><td>23.46</td><td>5.91</td><td>22.20</td><td>105.74</td><td>86.15</td><td>20.83</td><td>-1.66</td></tr><tr><td>RF Inv. [44]</td><td>27.99</td><td>6.74</td><td>20.20</td><td>179.73</td><td>139.85</td><td>21.71</td><td>4.34</td></tr><tr><td>RF Edit [53]</td><td>27.60</td><td>6.56</td><td>24.44</td><td>113.20</td><td>56.26</td><td>22.08</td><td>5.18</td></tr><tr><td>BrushEdit [26]</td><td>25.81</td><td>6.17</td><td>32.16</td><td>17.22</td><td>8.46</td><td>22.44</td><td>3.33</td></tr><tr><td>FLUX Fill [1]</td><td>25.76</td><td>6.31</td><td>32.53</td><td>25.59</td><td>8.55</td><td>22.40</td><td>5.71</td></tr><tr><td>Ours</td><td>27.21</td><td>6.49</td><td>35.87</td><td>9.92</td><td>4.69</td><td>22.39</td><td>5.63</td></tr><tr><td>+NS+RI</td><td>28.05</td><td>6.40</td><td>33.30</td><td>14.80</td><td>7.45</td><td>23.62</td><td>9.15</td></tr></table>

Table 2. Ablation study for object removal task. CLIP $\sin ^ { * }$ and $\mathrm { I R } ^ { * }$ represent alignment between source prompt and new image through CLIP [40] and Image Reward [55] to evaluate whether remove particular object from image. NS indicates there is no skip step during inversion. RI indicates the addition of reinitialization strategy. AM indicates that using attention mask during inversion.   

<table><tr><td rowspan="2">Method</td><td colspan="2">Image Quality</td></tr><tr><td>|HPS ×102 ↑ AS ↑|</td><td>Text Align |CLIP Sim ↓IR×10 *</td></tr><tr><td>KV Edit (ours)</td><td>26.76 6.49</td><td>25.50 6.87</td></tr><tr><td>+NS</td><td>26.93 6.37</td><td>25.05 3.17</td></tr><tr><td>+NS+AM</td><td>26.72 6.35</td><td>25.00 2.55</td></tr><tr><td>+NS+RI</td><td>26.73 6.34</td><td>24.82 0.22</td></tr><tr><td>+NS+AM+RI</td><td>26.51 6.28</td><td>24.90 0.90</td></tr></table>

Implementation Details. We implement our method based on FLUX.1-[dev] [1], following the same framework as other Rectified Flow-based methods [23, 44, 53]. We maintain consistent hyperparameters with FlowEdit [23], using 28 timesteps in total, skipping the last 4 timesteps ( $N = 2 4$ to reduce cumulative errors, and setting guidance values to 1.5 and 5.5 for inversion and denoising processes respectively. NS in tables and charts represent no skip step $N = 2 8$ ). Other baselines retain their default parameters or use previously published results. Unless otherwise specified, "Ours" in tables refers to the inversion-based KV-Edit without the two optional enhancement techniques proposed in Sec. 3.4. All experiments are conducted on two NVIDIA 3090 GPUs with 24GB memory. Metrics Following [20, 21, 26], we use seven metrics across three dimensions to evaluate our method. For image quality, we report HPSv2 [63] and aesthetic scores [45]. For background preservation, we measure PSNR [19], LPIPS [63], and MSE. For text-image alignment, we report CLIP score [40] and Image Reward [55]. Notably, while Image Reward was previously used for quality assessment, we found it particularly effective at measuring text-image alignment, providing negative scores for unedited images. Based on this observation, we also utilize Image Reward to evaluate the successful removal of objects.

# 4.2. Editing Results

We conduct experiments on PIE-Bench [21], categorizing editing tasks into three major types: removing, adding, and changing objects. For practical applications, these tasks prioritize background preservation and text alignment, followed by overall image quality assessment.

Quantitative Comparison. Sec. 4 presents quantitative results including baselines, our method, and our method with the reinitialization strategy. We exclude results with the attention mask strategy, as it shows improvements only in specific cases. Our method surpasses all others in Masked Region Preservation metrics. Notably, as shown in Fig. 6, methods with PSNR below 30 fail to maintain background consistency, producing results that merely resemble the original. RF-Inversion [44], despite obtaining high image quality scores, generates entirely different backgrounds. Our method achieves the third-best image quality, which has been higher than the original images, and perfectly preserving the background at the same time. With the reinitialization process, we achieve optimal text alignment scores, as the injected noise disrupts the original content, enabling more effective editing in certain cases (e.g., object removal and color change). Even compared to training-based inpainting methods [1, 26], our approach better preserves backgrounds while following user intentions.

![](images/7.jpg)  
Figure 7. Ablation study of different optional strategies on object removal task. From left to right, applying more strategies leads to stronger removal effect and the right is the best.

Qualitative Comparison. Fig. 6 demonstrates our method's performance against previous works across three different tasks. For removal tasks, the examples shown require both enhancement techniques proposed in Sec. 3.4. Previous training-free methods fail to preserve backgrounds, particularly Flow-Edit [23] which essentially generates new images despite high quality. Interestingly, training-based methods like BrushEdit [26] and FLUXFill [1] exhibit notable phenomena in certain cases (first and third rows in Fig. 6). BrushEdit [26], possibly limited by generative model capabilities, produces meaningless content. FLUX-Fill [1] sometimes misinterprets text prompts, generating unreasonable content like duplicate subjects. In contrast, our method demonstrates satisfactory results, successfully generating text-aligned content while preserving backgrounds, eliminating the traditional trade-off between background preservation and foreground editing.

# 4.3. Ablation Study

We conduct ablation studies to illustrate the impact of two enhancement strategies proposed in Sec. 3.4 and the no-skip step on our method's object removal performance. Tab. 2 presents the results in terms of image quality and text alignment scores. Notably, for text alignment evaluation, we compute the similarity between the generated results and the original prompt using CLIP [40] and Image Reward [55] models. This metric proves more discriminative in removal tasks, as still presenting of specific objects in the final images significantly increases the similarity scores. As shown in Tab. 2, the combination of NS (No-skip) and RI (Reinitialization) achieves the optimal text alignment scores. However, we observe a slight decrease in image quality metrics after incorporating these components. We attribute this phenomenon to the presence of too large masks in the benchmark, where no-skip, reinitialization, and attention mask collectively disrupt substantial information, leading to some discontinuities in the generated images. Consequently, these strategies should be viewed as optional enhancements for editing effects rather than universal solutions applicable to all scenarios.

Table 3. User Study. We compared our method with four popular baselines. Participants were asked to choose their preferred option or indicate if both methods were equally good or not good based on four criteria. We report the win rates of our method compared to baseline excluding equally good or not good instances. Random\* denotes the win rate of random choices.   

<table><tr><td>ours vs.</td><td>Quality↑</td><td>Background↑</td><td>Text↑</td><td>Overall↑</td></tr><tr><td>Random*</td><td>50.0%</td><td>50.0%</td><td>50.0%</td><td>50.0%</td></tr><tr><td>RF Inv. [44]</td><td>61.8%</td><td>94.8%</td><td>79.6%</td><td>85.1%</td></tr><tr><td>RF Edit [53]</td><td>54.5%</td><td>90.5%</td><td>75.0%</td><td>73.6%</td></tr><tr><td>BrushEdit [26]</td><td>71.8%</td><td>66.7%</td><td>68.7%</td><td>70.2%</td></tr><tr><td>FLUX Fill [1]</td><td>60.0%</td><td>53.7%</td><td>58.6%</td><td>61.9%</td></tr></table>

Fig. 7 visualizes the impact of these strategies. In the majority of cases, reinitialization alone suffices to achieve the desired results, while a small subset of cases requires additional attention masking for enhanced performance.

# 4.4. User Study

We conduct an extensive user study to compare our method with four baselines, including the training-free methods RFEdit [53], RF-Inversion [44], and the training-based methods BrushEdit [26] and Flux-Fill [1]. We use 110 images from the "random class" in the PIE-Bench [21] (excluding style transfer task, images without backgrounds, and controversial content). More than 20 participants are asked to compare each pair of methods based on four criteria: image quality, background preservation, text alignment, and overall satisfaction. As shown in Tab. 3, our method significantly outperforms the previous methods, even surpassing Flux-Fill [1], which is the official inpainting model of FLUX [1]. Additionally, users' feedback reveals that background preservation plays a crucial role in their final choices, even if RF-Edit [53] achieves high image quality but finally fails in satisfaction comparison.

# 5. Conclusion

In this paper, we introduce KV-Edit, a new training-free approach that achieves perfect background preservation in image editing by caching and reusing background keyvalue pairs. Our method effectively decouples foreground editing from background preservation through attention mechanisms in DiT, while optional enhancement strategies and memory-efficient implementation further improve its practical utility. Extensive experiments demonstrate that our approach surpasses both training-free methods and training-based inpainting models in terms of both background preservation and image quality. Moreover, we hope that this straightforward yet effective mechanism could inspire broader applications, such as video editing, multi-concept personalization, and other scenarios.

# References

[1] Flux. https://github.com/black-forestlabs/flux/. 2, 3, 6, 7, 8, 12, 13, 14   
[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 3   
[3] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. TOG, 42(4):111, 2023. 2   
[4] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel CohenOr.Stable flow: Vital layers for training-free image editing. arXiv preprint arXiv:2411.14430, 2024. 2, 5, 12   
[5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3   
[6] Sule Bai, Yong Liu, Yifei Han, Haoji Zhang, and Yansong Tang. Self-calibrated clip for training-free open-vocabulary segmentation. arXiv preprint arXiv:2411.15869, 2024. 3   
[7] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pages 1839218402, 2023. 2, 3   
[8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 33:1877 1901, 2020. 3   
[9] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, pages 2256022570, 2023. 2, 5, 6, 7   
10] Zhennan Chen, Yajie Li, Haofan Wang, Zhibo Chen, Zhengkai Jiang, Jun Li, Qian Wang, Jian Yang, and Ying Tai. Region-aware text-to-image generation via hard binding and soft refinement. arXiv preprint arXiv:2411.06558, 2024. 2   
11] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Real-time controllable motion generation via latent consistency model. In ECCV, pages 390408, 2024. 2   
12] Wenkai Dong, Song Xue, Xiaoyue Duan, and Shumin Han. Prompt tuning inversion for text-driven image editing using diffusion models. In ICCV, pages 74307440, 2023. 2, 3, 5, 14   
[13] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3   
[14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2   
[15] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 1600016009, 2022. 3   
[16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2, 3, 5, 6, 7, 14   
[17] Wenke Huang, Jian Liang, Zekun Shi, Didi Zhu, Guancheng Wan, He Li, Bo Du, Dacheng Tao, and Mang Ye. Learn from downstream and be yourself in multimodal large language model fine-tuning. arXiv preprint arXiv:2411.10928, 2024. 3   
[18] Xiaoke Huang, Jianfeng Wang, Yansong Tang, Zheng Zhang, Han Hu, Jiwen Lu, Lijuan Wang, and Zicheng Liu. Segment and caption anything. In CVPR, pages 13405 13417, 2024. 3   
[19] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 44(13):800801, 2008. 7   
[20] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In ECCV, pages 150168, 2024. 2, 3, 7   
[21] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In ICLR, 2024. 2, 6, 7, 8, 12, 14   
[22] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, pages 60076017, 2023. 2, 3   
[23] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024. 2, 6, 7, 8, 13   
[24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 1973019742, 2023. 3   
[25] Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang. Stylediffusion: Prompt-embedding inversion for text-based editing. arXiv preprint arXiv:2303.15649, 2023. 2   
[26] Yaowei Li, Yuxuan Bian, Xuan Ju, Zhaoyang Zhang, Ying Shan, and Qiang Xu. Brushedit: All-in-one image inpainting and editing. arXiv preprint arXiv:2412.10316, 2024. 2, 3, 6, 7,8   
[27] Haonan Lin, Mengmeng Wang, Jiahao Wang, Wenbin An, Yan Chen, Yong Liu, Feng Tian, Guang Dai, Jingdong Wang, and Qianying Wang. Schedule your edit: A simple yet effective diffusion noise schedule for image editing. arXiv preprint arXiv:2410.18756, 2024. 2   
[28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3   
[29] Aoyang Liu, Qingnan Fan, Shuai Qin, Hong Gu, and Yansong Tang. Lipe: Learning personalized identity prior for non-rigid image editing. arXiv preprint arXiv:2406.17236, 2024. 2   
[30] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024. 3   
[31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2024. 3   
[32] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022. 3, 4   
[33] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2022. 3, 4, 6   
[34] Yong Liu, Sule Bai, Guanbin Li, Yitong Wang, and Yansong Tang. Open-vocabulary segmentation with semantic-assisted calibration. In CVPR, pages 34913500, 2024. 3   
[35] Yong Liu, Cairong Zhang, Yitong Wang, Jiahao Wang, Yujiu Yang, and Yansong Tang. Universal segmentation at arbitrary granularity with language instruction. In CVPR, pages 34593469, 2024. 3   
[36] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2   
[37] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. arXiv preprint arXiv:2305.16807, 2023. 2   
[38] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, pages 60386047, 2023. 2   
[39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 41954205, 2023. 2, 4, 12, 15   
[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 87488763, 2021. 7, 8   
[41] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3   
[42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684 10695, 2022. 2   
[43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234241, 2015. 2, 3, 4   
[44] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 6, 7, 8   
[45] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 35:25278 25294, 2022. 7   
[46] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3, 4, 6   
[47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3   
[48] Siao Tang, Xin Wang, Hong Chen, Chaoyu Guan, Zewen Wu, Yansong Tang, and Wenwu Zhu. Post-training quantization with progressive calibration and activation relaxing for text-to-image diffusion models. In ECCV, pages 404420, 2024. 2   
[49] Yoad Tewel, Rinon Gal, Dvir Samuel, Yuval Atzmon, Lior Wolf, and Gal Chechik. Add-it: Training-free object insertion in images with pretrained diffusion models. arXiv preprint arXiv:2411.07232, 2024. 2, 5   
[50] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, pages 19211930, 2023. 2, 5, 14   
[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. page 60006010, 2017. 4   
[52] Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards accurate post-training quantization for diffusion models. In CVPR, pages 16026 16035, 2024. 2   
[53] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. 2, 6, 7, 8   
[54] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Effcient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023. 3   
[55] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. NeurIPS, 36:1590315935, 2023. 7, 8   
[56] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with language-guided diffusion models. In CVPR, pages 94529461, 2024. 2, 6, 7   
[57] Yu Xu, Fan Tang, Juan Cao, Yuxin Zhang, Xiaoyu Kong, Jintao Li, Oliver Deussen, and Tong-Yee Lee. Headrouter: A training-free image editing framework for mmdits by adaptively routing attention heads. arXiv preprint arXiv:2411.15034, 2024. 2   
[58] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In CVPR, pages 1815518165, 2022. 3   
[59] Zhao Yang, Jiaqi Wang, Xubing Ye, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Languageaware vision transformer for referring segmentation. TPAMI, 2024. 3   
[60] Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. Atp-llava: Adaptive token pruning for large vision language models. arXiv preprint arXiv:2412.00447, 2024. 3   
[61] Xubing Ye, Yukang Gan, Xiaoke Huang, Yixiao Ge, Ying Shan, and Yansong Tang. Voco-llama: Towards vision compression with large language models. arXiv preprint arXiv:2406.12275, 2024.   
[62] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams. arXiv preprint arXiv:2406.08085, 2024. 3   
[63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586595, 2018. 7   
[64] Yixuan Zhu, Wenliang Zhao, Ao Li, Yansong Tang, Jie Zhou, and Jiwen Lu. Flowie: Efficient image enhancement via rectified flow. In CVPR, pages 1322, 2024. 2   
[65] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In ECCV, pages 195211, 2024. 2, 3

# Appendix

In this supplementary material, we provide more details and findings. In Appendix A, we present additional experimental results and implementation details of our proposed KVEdit. Appendix B provides further discussion and data regarding our inversion-free methodology. Appendix C details the design and execution of our user study. Moreover, In Appendix D, we discuss potential future directions and current limitations of our work.

# A. Implementation and More Experiments

Implementation Details. Our code is built on Flux [1], with modifications to both double block and single block to incorporate KV cache through additional function parameters. Input masks are first downsampled using bilinear interpolation, then transformed from single-channel to 64- channel representations following the VAE in Flux [1]. In the feature space, the smallest pixel unit is 16 dimensions rather than the entire 64-dimensional token. Therefore, in addition to KV cache, we preserve the intermediate image features at each timestep to ensure fine-grained editing capabilities. In our experiment, inversion and denoising can be performed independently, allowing a single image to be inverted just once and then edited multiple times with different conditions, further enhancing the practicality of this workflow. Experimental Results. Due to space constraints in the main paper, we only present results on the PIE-Bench [21]. Here, we provide additional examples demonstrating the effectiveness of our approach. To further showcase the flexibility of our method, Fig. A and Fig. B present various editing target applied to the same source image, without explicitly labeling the input masks because each case corresponds to a different mask. Fig. D illustrates the impact of steps and reinitialization strategy on the color changing tasks and inpainting tasks. When changing colors, as the number of skip-steps decreases and reinitialization strategy is applied, the color information in the tokens is progressively disrupted, ultimately achieving successful results. In our experiments, the optimal number of steps to skip depends on image resolution and content, which can be adjusted based on specific needs and feedback. Unlike previous training-free methods, our approach even can be applied to inpainting tasks after employing reinitialization strategy, as demonstrated in the third row of Fig. D. The originally removed regions in inpainting tasks can be considered as black objects, thus requiring reinitialization strategy to eliminate pure black information and generate meaningful content. We plan to further extend our method to inpainting tasks in future work, as there are currently very few training-free methods available for this application. Attention Scale When dealing with large masks (e.g., background changing tasks), our original method may produce discontinuous images including conflicting content, as illustrated in Fig. C. Stable-Flow [4] demonstrated that during image generation with DiT [39], image tokens primarily attend to their local neighborhood rather than globally across most layers and timesteps.

![](images/8.jpg)  
.

![](images/9.jpg)  
.

Consequently, although our approach treats the background as a condition to guide new content generation, large masks can introduce generation bias which ignore existing content and generate another objects. Based on this analysis, we propose a potential solution as shown in Fig. C. We directly increase the attention weights from masked regions to unmasked regions in the attention map (produced by query-key multiplication), effectively mitigating the bias impact. This attention scale mechanism enhances content coherence by strengthening the influence of preserved background on new content.

# B. More Discussions on Inversion-Free

We implement inversion-free editing on Flux [1] based on the code provided by FlowEdit [23]. As noted in FlowEdit [23], adding random noise at each editing step may introduce artifacts, a phenomenon we also demonstrate in the main paper. In this section, we primarily explore the impact of inversion-free methods on memory consumption.

![](images/10.jpg)  
Figure C. Implementation of attention scale. The scale can be adjusted to achieve optimal results.

![](images/11.jpg)  
Figure D. Additional ablation studies on two tasks. The first and second rows demonstrate the impact of timesteps and reinitialization strategy $\mathbf { \Pi } ( \mathbf { R I } )$ on color changing. The third row demonstrates the impact of timesteps and RI on the inpainting tasks.

Algorithm A demonstrates the implementation of inversion-free KV-Edit, where "KV-inversion" and "KVdenoising" refer to single-step noise prediction with KV cache. KV cache is saved during a one-time inversion process and immediately utilized in the denoising process. The final vector can be directly added to the original image without first inversing it to noise. This strategy ensures that the space complexity of KV cache remains $O ( 1 )$ along the time dimension. Moreover, resolution has a more significant impact on memory consumption as the number of image tokens grows at a rate of $O ( n ^ { 2 } )$ . We conducted experiments across various resolutions and time steps, reporting memory usage in Tab. A. When processing high-resolution images and more timesteps, personal computers struggle to accommodate the mem Table A. Memory usage at different resolutions and timesteps. Our approach has a space complexity of $O ( n )$ along the time dimension, while inversion-free methods achieve $O ( 1 )$ .   

<table><tr><td rowspan="2">timesteps</td><td colspan="2">512 × 512</td><td colspan="2">768 × 768</td></tr><tr><td>Ours</td><td>+Inf.</td><td>Ours</td><td>+Inf.</td></tr><tr><td>24 steps</td><td>16.2G</td><td>1.9G</td><td>65.8G</td><td>3.5G</td></tr><tr><td>28 steps</td><td>19.4G</td><td>1.9G</td><td>75.6G</td><td>3.5G</td></tr><tr><td>32 steps</td><td>22.1G</td><td>1.9G</td><td>86.5G</td><td>3.5G</td></tr></table>

# Algorithm A Simplified Inf. version KV-Edit

1: Input: $t _ { i }$ , real image $x _ { 0 } ^ { s r c }$ , foreground $z _ { t _ { i } } ^ { f g }$ ,foreground   
region mask, KV cache $C$   
2: Output: Prediction vector V   
3: $N _ { t _ { i } } \sim \mathcal { N } ( 0 , 1 )$   
4: $x _ { t _ { i } } ^ { s r c } = ( 1 - t _ { i } ) x _ { t _ { 0 } } ^ { s r c } + t _ { i } N _ { t _ { i } }$ b   
5: 6: $V _ { \theta t _ { i } } ^ { s r c } , C = \mathrm { K V - I n v e r i s o n } ( x _ { t _ { i } } ^ { s r c } , t _ { i } , C )$ $\widetilde { z } _ { t _ { i } } ^ { f g } = z _ { t _ { i } } ^ { f g } + m a s k \cdot ( x _ { t _ { i } } ^ { s r c } - x _ { 0 } ^ { s r c } )$ $\widetilde { V } _ { \theta t _ { i } } ^ { f g }$ $C = \mathrm { K V - D e n o s i n g } ( \widetilde { z } _ { t _ { i } } ^ { f g } , t _ { i } , C )$   
8: Return $V _ { \theta t _ { i } } ^ { f g } = \widetilde { V } _ { \theta t _ { i } } ^ { f g } - V _ { \theta t _ { i } } ^ { s r c }$

![](images/12.jpg)  
1In terms of image quality and aesthetics, which image is better, A or B? Option C cannot be selected. OA OB OC   
Figure E. User study. We provide a sample where participants were presented with the original image, editing prompts, results from two different methods for comparison and four questions from four aspects.

Rs h  oy good, choose C. B  h ll. OA OB OC dissatisfied with both, choose C. ory requirements. Nevertheless, we still recommend the inversion-based KV-Edit approach for several reasons: 1. Current inversion-free methods occasionally introduce artifacts. 2. Inversion-based KV-Edit enables multiple editing attempts after a single inversion, significantly improving usability and workflow efficiency. 3. Large generative models inherently require substantial GPU memory, which presents another challenge for personal computers. Therefore, we position inversion-based KV-Edit as a server-side technology.

# C. User Study Details

We conduct our user study in a questionnaire format to collect user preferences for different methods. We observe that in most cases, users struggle to distinguish the background effects of training-based inpainting methods (e.g., FLUX-Fill [1] sometimes increases grayscale tones in images). Therefore, we allowed participants to select "equally good" regarding background quality. Additionally, PIE-Bench [21] contains several challenging cases where all methods fail to complete the editing tasks satisfactorily. Consequently, we allow users to select "neither is good" for text alignment and overall satisfaction metrics, as illustrated in Fig. E. We implement a single-blind mechanism where the corresponding method for each question is randomly sampled, ensuring fairness in the comparison. We collect over 2,000 comparison results and calculate our method's win rate after excluding cases where both methods are rated equally.

# D. Limitations and Future Work

In this section, we outline the current challenges faced by our method and potential future improvements. While our approach effectively preserves background content, it struggles to maintain foreground details. As shown in Fig. D, when editing garment colors, clothing appearance features may be lost, such as the style, print or pleats. Typically, during the generation process, early steps determine the object's outline and color, with specific details and appearance emerging later. In the contrast, during inversion, customized object details are disrupted first and subsequently influenced by new content during denoising. This represents a common challenge in the inversion-denoising paradigm [12, 16, 50]. In future work, we could employ trainable tokens to preserve desired appearance information during inversion and inject it during denoising, still without fine-tuning of the base generative model. Furthermore, our method could be adapted to other modalities, such as video and audio editing, image inpainting tasks. We hope that "KV cache for editing" can be considered an inherent feature of the DiT [39] architecture.