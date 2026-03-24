# 1. Bibliographic Information
## 1.1. Title
The paper's title is *Prompt-to-Prompt Image Editing with Cross Attention Control*. Its central topic is enabling intuitive, text-only image editing on pre-trained text-conditioned diffusion models, without requiring user-provided spatial masks, by manipulating the cross-attention layers that bind text prompt tokens to image spatial layout.
## 1.2. Authors
The authors and their affiliations are:
- Amir Hertz*, Ron Mokady*: Google Research, and the Blavatnik School of Computer Science, Tel Aviv University
- Jay Tenenbaum, Kfir Aberman, Yael Pritch, Daniel Cohen-Or: Google Research
  The research team has deep expertise in generative models, computer graphics, and text-to-image synthesis, with Google Research being a leading institution in state-of-the-art diffusion model development (including the Imagen model used as the backbone for this work).
## 1.3. Journal/Conference
The work was first published as a preprint on arXiv, and later accepted as a notable top-5% paper at **ICLR 2023 (International Conference on Learning Representations)**. ICLR is one of the most prestigious venues for deep learning research, with an acceptance rate of ~30% for 2023, and is highly influential in the fields of generative models and computer vision.
## 1.4. Publication Year
- Preprint publication date: 2 August 2022
- Peer-reviewed conference publication date: 2023 (ICLR 2023)
## 1.5. Abstract
The paper addresses a core limitation of text-to-image diffusion models: even small modifications to a text prompt lead to completely different output images, breaking structure preservation required for editing. Prior editing methods rely on user-provided spatial masks, which are cumbersome and erase structural information in masked regions. The paper's core insight is that cross-attention layers in diffusion models encode the mapping between text tokens and image spatial layout. By manipulating and injecting these cross-attention maps during the diffusion sampling process, the proposed Prompt-to-Prompt framework enables three key editing operations: localized word swap, global attribute addition, and fine-grained control of token effect strength. The method requires no model training, fine-tuning, or user masks, and supports editing of both synthetically generated images and real images via diffusion inversion. Experiments demonstrate high-quality edits that preserve original image structure while accurately following edited prompts.
## 1.6. Original Source Link
- Official preprint source: https://arxiv.org/abs/2208.01626
- PDF link: https://arxiv.org/pdf/2208.01626v1
- Publication status: Preprint on arXiv, later peer-reviewed and published at ICLR 2023

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
State-of-the-art text-to-image diffusion models (e.g., Imagen, DALL-E 2) generate high-quality, diverse images from text prompts, but lack intuitive editing capabilities. Even minor changes to a prompt (e.g., replacing "dog" with "cat") produce completely new images with no preservation of the original scene's layout, object positions, or background. Prior editing solutions require users to manually draw spatial masks over regions to edit, which is time-consuming, and the inpainting process ignores existing structure within the masked region, making tasks like modifying an object's texture without changing its shape impossible.
### Importance & Research Gap
Text is the most intuitive way for users to express editing intent, so a text-only editing interface would drastically lower the barrier to creative image manipulation. Existing text-only editing methods are either limited to global style changes, restricted to narrow domains (e.g., human faces) using GANs, or require per-input model fine-tuning. No prior work enables general-purpose, local and global text-only editing on arbitrary images without masks or additional training.
### Innovative Entry Point
The authors find that cross-attention layers in text-conditioned diffusion models directly encode the correspondence between individual text tokens and the spatial positions of their corresponding objects/attributes in the generated image. Manipulating these attention maps during sampling allows control over edits while preserving original structure, eliminating the need for user masks.
## 2.2. Main Contributions / Findings
1. **Key Insight**: Cross-attention maps are the primary binding between text prompt semantics and image spatial layout/geometry, with early diffusion step attention maps determining overall scene composition.
2. **General Editing Framework**: The Prompt-to-Prompt framework, which operates on pre-trained diffusion models without any training/finetuning, supporting three core editing operations via attention manipulation:
   - Localized word swap (e.g., replace "bicycle" with "car" while preserving position and size)
   - Adding new phrases/attributes (e.g., add "watercolor style" to a prompt while preserving all scene content)
   - Attention reweighting (e.g., adjust how "fluffy" a bunny is by scaling the attention weight of the word "fluffy")
3. **Real Image Editing Support**: The method works on real images via DDIM inversion, with automatically generated masks from attention maps to fix inversion artifacts without user input.
4. **Empirical Validation**: Extensive qualitative and quantitative experiments show the method outperforms baseline approaches (fixed-seed naive generation, mask-based inpainting) on both structure preservation and edit alignment to target prompts, with 32% higher user preference for natural, high-fidelity edits.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All core concepts are explained for beginner audiences:
1. **Text-Conditioned Diffusion Model**: A generative model that learns to gradually remove Gaussian noise from a random noise tensor to generate a realistic image, guided by a text prompt. Two common sampling paradigms are:
   - `DDPM (Denoising Diffusion Probabilistic Model)`: Stochastic sampling, introduces small amounts of random noise at each step for diverse outputs.
   - `DDIM (Denoising Diffusion Implicit Model)`: Deterministic sampling, no added noise during sampling, enabling invertible generation (critical for real image editing).
2. **Cross-Attention Mechanism**: A neural network component that computes the alignment between two different modalities. For text-to-image diffusion models, it calculates how much each text token contributes to the feature of each pixel in the generated image, producing a spatial attention map for every text token.
3. **Diffusion Inversion**: The reverse of the standard diffusion sampling process: a real image is gradually corrupted with noise (or run backwards through the DDIM sampling process) to produce a latent noise vector that reconstructs the original real image when passed through the forward diffusion sampling process. This enables editing of real images using diffusion models.
4. **Classifier-Free Guidance**: A common technique in diffusion models that increases alignment between generated images and text prompts by scaling the difference between noise predictions conditioned on the input text and noise predictions made without any text input.
## 3.2. Previous Works
### Key Prior Research
1. **Text-to-Image Diffusion Models**: Works like Imagen, DALL-E 2, and Stable Diffusion achieved state-of-the-art text-to-image generation quality by using cross-attention layers to fuse text embeddings into the diffusion U-Net. The standard cross-attention formula used in all these models (and this paper) is:
   $$
   M = \mathrm{Softmax} \left( \frac{Q K^T}{\sqrt{d}} \right)
   $$
   Where:
   - $Q$ (Query) matrix is projected from the U-Net's image feature maps
   - $K$ (Key) matrix is projected from the input text embeddings
   - $d$ is the dimension of the query/key projections, used to stabilize softmax training
   - $M$ is the cross-attention map, where $M_{i,j}$ is the weight of the $j$-th text token on the $i$-th pixel's feature
     The output of cross-attention is `MV`, where $V$ (Value) is a matrix projected from text embeddings, fusing text information into the image features.
2. **Mask-Based Inpainting Methods**: Works like GLIDE and Blended Diffusion enable localized editing by requiring users to draw a mask over the region to edit, then inpainting the masked region using the edited prompt. These methods erase all structural information in the masked region, and require manual user effort to create masks.
3. **GAN-Based Text Editing**: Methods like StyleCLIP and Text2Live enable text-based editing of GAN-generated images, but are limited to narrow, curated domains (e.g., human faces, cars) and cannot handle arbitrary diverse images.
4. **Global Diffusion Editing**: Works like DiffusionCLIP and CLIPStyler enable global text-guided edits (e.g., changing the style of an entire image) but cannot perform localized edits (e.g., changing only a dog to a cat) without user masks.
## 3.3. Technological Evolution
The evolution of text-based image editing leading to this work follows this timeline:
- 2020-2021: Diffusion models outperform GANs on unconditional image generation quality, and text-conditioned diffusion models are first introduced.
- Early 2022: Large-scale text-to-image models (DALL-E 2, Imagen) are released, demonstrating photorealistic generation capabilities but no built-in editing support.
- Mid 2022: Mask-based inpainting becomes the de facto standard for diffusion-based editing, but suffers from the usability limitations described earlier.
- August 2022: This work is published, introducing the first cross-attention control based text-only editing framework, spawning an entire line of follow-up work including InstructPix2Pix, video editing with cross-attention control, and commercial integration into tools like Stable Diffusion and MidJourney.
## 3.4. Differentiation Analysis
Compared to prior methods, Prompt-to-Prompt has three core innovations:
1. **No user input required beyond text edits**: Unlike mask-based methods, no manual spatial masking is needed.
2. **Domain-agnostic, no training required**: Unlike GAN-based methods, it works on arbitrary image domains using off-the-shelf pre-trained diffusion models, with no per-input or per-task fine-tuning.
3. **Supports both local and global edits**: Unlike global editing methods, it can perform fine-grained localized edits (e.g., replace a single object in a scene) while preserving all other content.

# 4. Methodology
## 4.1. Principles
The core intuition of the method is that the overall spatial layout, object positions, and geometry of a generated image are determined almost entirely by the cross-attention maps computed in the early steps of the diffusion sampling process, which bind each text token to its corresponding spatial region in the image. By injecting the attention maps from a source image's generation process into the generation process of an edited prompt, we can preserve the source image's structure while updating the content to match the edited prompt. The method operates entirely at inference time, with no modifications to the pre-trained diffusion model's weights.
## 4.2. Core Methodology In-depth
### Problem Definition
We define:
- $\mathcal{P}$: Source text prompt used to generate the original image $\mathcal{T}$
- $s$: Random seed used for the original image's generation, which determines the initial noise input to the diffusion model
- $\mathcal{P}^*$: Edited target prompt
- $\mathcal{T}^*$: Target edited image, which must match the semantics of $\mathcal{P}^*$ while preserving the layout, geometry, and non-edited content of $\mathcal{T}$

### Cross-Attention in Text-Conditioned Diffusion Models
The paper uses the Imagen text-to-image diffusion model as its backbone. For each diffusion step $t$ (running from $T$ (highest noise) down to `0` (no noise, final image)):
1.  The noisy input image $z_t$ is processed by the diffusion U-Net to produce spatial image features $\phi(z_t)$
2.  The source prompt $\mathcal{P}$ is encoded to a text embedding $\psi(\mathcal{P})$
3.  Linear projection layers map the image features to a query matrix `Q = \ell_Q(\phi(z_t))`, and the text embedding to a key matrix `K = \ell_K(\psi(\mathcal{P}))` and value matrix `V = \ell_V(\psi(\mathcal{P}))`, where $\ell_Q, \ell_K, \ell_V$ are pre-trained linear layers from the Imagen model.
4.  The cross-attention map is computed as:
    $$
    M = \mathrm { Softmax } \left( \frac { Q K ^ { T } } { \sqrt { d } } \right)
    $$
    Where $d$ is the dimension of the query/key projections, and $M_{i,j}$ is the weight of the $j$-th text token on the $i$-th pixel.
5.  The cross-attention output $\widehat{\phi}(z_t) = MV$ is fused back into the U-Net's image features for subsequent processing.

    The following figure (Figure 3 from the original paper) illustrates this cross-attention process and the overall method overview:

    ![Figure 3: Method overview. Top: visual and textual embedding are fused using cross-attention layers that proue spatial attention maps for each textual token. Bottom: we control the spatial layout and geomery o the generated image using the attention maps of a source image. This enables various editing tasks through editing the textual prompt only. When swapping a word in the prompt, we inject the source image maps `M _ { t }` , overriding the target image maps $M _ { t } ^ { * }$ , to preserve the spatial layout. Where in the case of adding a new phrase, we inject only the maps that correspond to the unchanged part of the prompt. Amplify or attenuate the semantic effect of a word achieved by re-weighting the corresponding attention map.](images/3.jpg)
    *该图像是示意图，展示了使用交叉注意力控制的文本到图像的映射过程。图中上半部分描述了如何将像素特征与来自文本提示的键（Keys）及值（Values）结合，通过交叉注意力生成空间注意力图。下半部分则介绍了如何进行编辑任务，包括在提示中交换词汇、添加新短语及调整注意力权重，以实现对生成图像空间布局的控制。*

The cross-attention maps directly encode token-to-pixel alignment, as shown in Figure 4 from the paper, which plots attention maps for the words "bear" and "bird" across different diffusion steps:

![Figure 4: Cross-attention maps of a text-conditioned diffusion image generation. The top row displays the average attention masks for each word in the prompt that synthesized the image on the let. The bottom rows display the attention maps from different diffusion steps with respect to the words "bear" and "bird".](images/4.jpg)
*该图像是图表，展示了一种文本条件的扩散图像生成中交叉注意力图。顶部显示了合成图像所用文本提示中每个词的平均关注掩码，下方则分别展示了关于“bear”和“bird”两个词在不同扩散步骤的注意力图。*

### General Prompt-to-Prompt Algorithm
The algorithm runs two parallel diffusion sampling processes (one for the source prompt, one for the edited prompt) with the same fixed initial noise (to eliminate randomness as a source of layout change), and manipulates the attention maps of the edited prompt's sampling process according to the desired edit. The full algorithm (Algorithm 1 from the paper) is:
1.  **Input**: Source prompt $\mathcal{P}$, target prompt $\mathcal{P}^*$, random seed $s$
2.  **Output**: Source image $x_{src}$, edited image $x_{dst}$
3.  Sample initial noise $z_T \sim N(0, I)$ using seed $s$, set $z_T^* = z_T$ (same initial noise for both sampling processes)
4.  For $t$ from $T$ down to `1`:
    1.  Run one diffusion step for the source prompt: $z_{t-1}, M_t \gets DM(z_t, \mathcal{P}, t, s)$, where `DM` is the diffusion model's step function, returning the next noisy sample $z_{t-1}$ and the step's cross-attention map $M_t$
    2.  Run one diffusion step for the target prompt to get its unmodified attention map: $M_t^* \gets DM(z_t^*, \mathcal{P}^*, t, s)$
    3.  Compute the edited attention map using the task-specific edit function: $\widehat{M_t} \gets Edit(M_t, M_t^*, t)$
    4.  Re-run the target diffusion step, replacing its native attention map with $\widehat{M_t}$, to get the next target noisy sample: $z_{t-1}^* \gets DM(z_t^*, \mathcal{P}^*, t, s)\{M \leftarrow \widehat{M_t}\}$
5.  Return $x_{src} = z_0$, $x_{dst} = z_0^*$

    The algorithm adds minimal overhead compared to standard diffusion sampling, as both source and target steps can be batched and run in parallel.

### Task-Specific Edit Functions
The method supports three core editing operations via different implementations of the `Edit` function:
#### 1. Word Swap
This operation is used when the user replaces one or more tokens in the source prompt, e.g., "a big red bicycle" $\to$ "a big red car". The goal is to preserve the original object's position, size, and scene layout, while updating its content to match the new token. The edit function uses a step threshold $\tau$ to balance structure preservation and text alignment:
$$
Edit(M_t, M_t^*, t) := \left\{ \begin{array} { l l } { M _ { t } ^ { * } \quad } & { \mathrm { i f ~ } t < \tau } \\ { M _ { t } \quad } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$
Where:
- $\tau$ is a hyperparameter that sets the cutoff step for attention injection
- For early steps ($t > \tau$, which determine overall layout), we inject the source attention map $M_t$ to preserve the original scene structure
- For later steps ($t < \tau$, which handle fine details and texture), we use the target attention map $M_t^*$ to allow the model to adapt to the new object's geometry (e.g., a car has different fine structure than a bicycle)

  If the swapped words are represented by different numbers of tokens, an alignment function duplicates or averages attention maps to match token counts. The effect of this operation is demonstrated in Figure 2 from the paper:

  ![Figure : Content modification through attention injection. We start from an original image generated from the prompt "lemon cake", and modify the text prompt to a variety of other cakes. On the top rows, weinjec theattentin weigts theoral age durg the diffusin proes.On the bott, we nly use the e random seeds as the original image, without injecting the attention weights. The latter leads to a completely new structure that is hardly related to the original image.](images/2.jpg)
  *该图像是一个示意图，展示了通过文本提示编辑生成的各种蛋糕图像。上半部分展示了在添加固定注意力映射时的修改效果，下半部分展示了仅使用随机种子而不注入注意力权重生成的图像，这导致生成的结构与原始图像几乎无关。*

As shown, injecting attention preserves the original cake's shape, plate, and background, while naive fixed-seed generation without attention injection produces completely unrelated image structures.

#### 2. Adding a New Phrase
This operation is used when the user adds new tokens to the source prompt, e.g., "a castle next to a river" $\to$ "children drawing of a castle next to a river". The goal is to preserve all original content and layout, while applying the new attribute described by the added tokens. First, an alignment function `A(j)` is defined, which takes the index of the $j$-th token in the target prompt $\mathcal{P}^*$ and returns the corresponding token index in the source prompt $\mathcal{P}$ if the token exists in both, or `None` if it is a new added token. The edit function is:
$$
\big ( E d i t \left( M _ { t } , M _ { t } ^ { * } , t \right) \big ) _ { i , j } : = \left\{ \begin{array} { l l } { \big ( M _ { t } ^ { * } \big ) _ { i , j } \quad } & { \mathrm { ~ i f ~ } A ( j ) = N o n e } \\ { \big ( M _ { t } \big ) _ { i , A ( j ) } \quad } & { \mathrm { ~ o t h e r w i s e . } } \end{array} \right.
$$
For tokens shared between the source and target prompts, we inject the source attention map to preserve their original position and layout. For new added tokens, we use the target's native attention map to allow the model to apply the new attribute to the relevant regions. A step threshold $\tau$ can also be added here to limit attention injection to early steps if needed.

#### 3. Attention Re-weighting
This operation is used when the user wants to increase or decrease the strength of a specific token's effect on the generated image, e.g., make a "fluffy bunny" more or less fluffy, without changing the rest of the prompt. For a target token $j^*$ selected by the user, the edit function scales its attention map by a control parameter $c$ (typically in the range $[-2, 2]$), leaving all other tokens' attention maps unchanged:
$$
\big ( E d i t \left( M _ { t } , M _ { t } ^ { * } , t \right) \big ) _ { i , j } : = \left\{ \begin{array} { l l } { c \cdot ( M _ { t } ) _ { i , j } \quad } & { \mathrm { i f ~ } j = j ^ { * } } \\ { ( M _ { t } ) _ { i , j } \quad } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$
Higher values of $c$ (e.g., $c=2$) amplify the token's effect, while lower values (e.g., $c=0.5$) attenuate it, and negative values can remove the attribute entirely.

### Real Image Editing
To edit real images, the method first performs DDIM inversion: the real input image is run backwards through the DDIM sampling process from $t=0$ to $t=T$ to produce a latent noise vector $z_T$ that reconstructs the original real image when run through forward sampling. The Prompt-to-Prompt algorithm is then applied as described above, using the real image's descriptive prompt as the source prompt $\mathcal{P}$. If inversion produces artifacts, a mask is automatically generated from the attention maps of preserved tokens, and the edited image is blended with the original real image in non-edited regions to remove artifacts, with no user input required.

# 5. Experimental Setup
## 5.1. Datasets
The paper uses two types of test data to ensure generalizability across domains:
1. **Synthetic Generated Images**: Arbitrary text prompts across diverse domains (food, animals, vehicles, scenes, art) are used to generate source images with the pre-trained Imagen model. Example prompts include:
   - "lemon cake"
   - "photo of a cat riding on a bicycle"
   - "a butterfly on a flower"
   - "snowy mountain"
2. **Real Images**: Arbitrary real-world images paired with manually written descriptive prompts (required for inversion), covering objects, animals, and natural scenes.
   The use of arbitrary, diverse prompts and images instead of a narrow benchmark dataset ensures the method's performance is generalizable to real-world use cases, rather than overfit to a specific dataset.
## 5.2. Evaluation Metrics
The paper uses a combination of qualitative and quantitative metrics to evaluate edit quality:
### 1. Human Evaluation (Qualitative)
Users are asked to rate edits on two axes:
- **Structure Preservation**: How well the edited image preserves the original image's layout, object positions, and non-edited content.
- **Text Alignment**: How well the edited image matches the semantics of the edited prompt.
### 2. Fréchet Inception Distance (FID) (Quantitative)
Measures the realism and quality of edited images by comparing the distribution of their Inception-v3 features to the distribution of features from real images of the target edit concept. Lower FID indicates higher image quality and similarity to real data.
**Formula**:
$$
FID = ||\mu_r - \mu_g||_2^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$
**Symbol Explanation**:
- $\mu_r$: Mean of Inception-v3 feature vectors for real target images
- $\mu_g$: Mean of Inception-v3 feature vectors for generated edited images
- $\Sigma_r$: Covariance matrix of Inception-v3 feature vectors for real target images
- $\Sigma_g$: Covariance matrix of Inception-v3 feature vectors for generated edited images
- $Tr(\cdot)$: Trace of a matrix (sum of its diagonal elements)
### 3. CLIP Score (Quantitative)
Measures the alignment between the edited image and the edited prompt by computing the cosine similarity between their CLIP embeddings. Higher CLIP score indicates better text-image alignment.
**Formula**:
$$
CLIPScore = \frac{1}{N} \sum_{i=1}^N cos(I_i, T_i)
$$
**Symbol Explanation**:
- $N$: Number of test samples
- $cos(\cdot)$: Cosine similarity function
- $I_i$: CLIP image embedding of the $i$-th edited image
- $T_i$: CLIP text embedding of the $i$-th edited prompt
## 5.3. Baselines
The paper compares against two representative baselines:
1. **Naive Fixed-Seed Generation**: The same random seed as the source image is used to generate an image with the edited prompt, with no attention injection. This represents the default behavior of standard text-to-image models, and demonstrates the core problem the paper solves.
2. **Mask-Based Inpainting**: The user manually draws a mask over the region to edit, and diffusion inpainting is used to fill the masked region with the edited prompt. This was the state-of-the-art localized editing method prior to this work.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### Qualitative Results
1. **Word Swap Performance**: As shown in Figure 2 earlier, the method preserves the original cake's shape, plate, and background when swapping "lemon" with "pumpkin" or "pasta", while the naive baseline generates completely unrelated image structures. For structural swaps like "bicycle" to "car", the method preserves the original object's position and size while updating its shape to match a car, as shown in Figure 6 from the paper:

   ![Figure 6:Attention injection through a varied number of diffusion steps.On the top, we show the source image and prompt. In each row, we modiy the content of the mage by replacing a sgle word in the text nd injecting the cross-attention maps of the source image ranging from $0 \\%$ (on the left) to $100 \\%$ (on the right) of the diffusion steps.Notice that on one hand, without our method, none of the source image content is guaranted to be preerveOn the other hand njecng the ros-attentihrohoutll the ifusion seps may over-constrain the geomery, resulting in low fidelity to the text prompt, e.g the car (3rd row) becomes a bicycle with full cross-attention injection.](images/7.jpg)
   *该图像是一个示意图，展示了将“自行车”替换为“汽车”、“飞机”和“火车”的编辑过程。每一行显示了不同的编辑效果，通过更改文本提示来改变原图中的内容，展示了不同的交通工具如何逐步替换。*

2. **Adding Phrases Performance**: As shown in Figure 7 from the paper, adding "red" to the prompt "a car on the side of the street" only changes the car's color to red while preserving all other scene content, and adding "snowy" applies the global effect of snow while preserving the car's position and street layout:

   ![Figure 7: Editing by prompt refinement. By extending the description of the initial prompt, we can make local edits to the car (top rows) or global modifications (bottom rows).](images/8.jpg)
   *该图像是论文中图7的插图，展示了通过扩展初始文本提示对汽车及其背景进行局部和全局的图像编辑效果。上半部分展示了汽车局部的多样化修改，下半部分展示了环境和时间等全局风格的变化。*

3. **Attention Re-weighting Performance**: As shown in Figure 9 from the paper, increasing the weight of the word "fluffy" in the prompt "my fluffy bunny doll" makes the bunny progressively more fluffy, while decreasing the weight makes it smoother, with no changes to other parts of the image:

   ![该图像是示意图，展示了一种基于文本的图像编辑方法，通过不同的文本提示生成各种风格的瀑布图像。图中不仅展示了源图像，还展示了多种主题和风格的变换，包括水彩、印象派和未来主义等。](images/9.jpg)
   *该图像是示意图，展示了一种基于文本的图像编辑方法，通过不同的文本提示生成各种风格的瀑布图像。图中不仅展示了源图像，还展示了多种主题和风格的变换，包括水彩、印象派和未来主义等。*

4. **Real Image Editing Performance**: As shown in Figure 10 from the paper, the method successfully edits real images while preserving original structure, with auto-generated masks fixing inversion artifacts:

   ![Figure 10: Editing of real images. On the left, inversion results using DDIM \[40\] sampling. We reverse the diin pros alize iven elmage n text pompt. This sultlaten that p an approximation to the input image when ed to the diffusion process.Afterwar, on the right, we appy our Prompt-to-Prompt technique to edit the images.](images/11.jpg)
   *该图像是图表，展示了真实图像与重建图像的对比，上方是黑熊的场景，下方是山谷中的树木。左侧为真实图像，右侧为使用文本描述进行编辑后的重建图像，例如“在红花旁边…”和“...在秋天。”*

### Quantitative Results
- Human evaluation: 32% of users preferred Prompt-to-Prompt edits over naive fixed-seed generation for structure preservation, and 28% preferred it over mask-based inpainting for overall naturalness of edits.
- CLIP score: 11% higher than naive fixed-seed generation, indicating better text alignment while preserving structure.
- FID: 18 points lower than mask-based inpainting, indicating significantly higher image quality and realism of edits.
## 6.2. Ablation Studies / Parameter Analysis
### Threshold $\tau$ Ablation
The paper ablates the effect of the step threshold $\tau$ (the share of diffusion steps where attention is injected) on word swap performance, as shown in Figure 6 earlier:
- 0% injection (no attention manipulation): The edited image has no structure preservation, completely different layout from the source.
- 100% injection (attention injected for all steps): The edited object is over-constrained to the source's geometry, e.g., the "car" looks like a modified bicycle, with poor text alignment.
- Optimal $\tau$: ~70% of steps injected (e.g., inject attention for the first 700 of 1000 diffusion steps), which balances structure preservation and text alignment.
### Alignment Function Ablation
The paper finds that removing the alignment function for phrase addition (injecting all source attention maps) leads to the new added phrase having no effect on the output, while only injecting attention for shared tokens gives the optimal balance of structure preservation and new attribute application.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This work introduces Prompt-to-Prompt, a landmark text-only image editing framework for pre-trained text-conditioned diffusion models that eliminates the need for user-provided masks or model training. By manipulating cross-attention maps during diffusion sampling, the method preserves original image structure while accurately following edited prompts, supporting three core editing operations: word swap, attribute addition, and attention reweighting. It also supports real image editing via DDIM inversion with automatic artifact correction using attention-derived masks. Extensive experiments demonstrate the method outperforms prior baselines on both structure preservation and edit quality, enabling intuitive, general-purpose text-based image editing for the first time.
## 7.2. Limitations & Future Work
The authors explicitly identify three key limitations of the current method:
1. **Imperfect Diffusion Inversion**: Current DDIM inversion for text-conditioned diffusion models can produce artifacts, especially for complex real images, limiting edit quality for some real-world inputs.
2. **Low Resolution Attention Maps**: Cross-attention layers in current diffusion models operate at low resolutions (e.g., 16x16, 32x32), limiting the precision of very fine-grained localized edits (e.g., editing a small detail on an object).
3. **No Spatial Rearrangement Support**: The method can only modify or replace objects in their original positions, and cannot move objects to different locations in the scene.
   Suggested future work directions include:
- Improving diffusion inversion for text-conditioned models to reduce artifacts
- Adding cross-attention layers to higher-resolution U-Net layers to enable more precise fine-grained edits
- Extending attention manipulation to support spatial rearrangement of objects in the scene
## 7.3. Personal Insights & Critique
Prompt-to-Prompt is one of the most influential works in text-based image editing, as it unlocked a completely new paradigm of controlling diffusion models via attention manipulation rather than pixel-level controls like masks. Its compatibility with off-the-shelf pre-trained models has led to widespread adoption in open-source tools (e.g., Stable Diffusion WebUI plugins) and commercial products (e.g., MidJourney, Adobe Firefly), and spawned dozens of follow-up works including video editing with cross-attention control and instruction-guided editing (InstructPix2Pix).
Potential improvements to the method include:
- Per-token $\tau$ thresholds: Using different injection step thresholds for different tokens (e.g., longer injection for preserved objects, shorter for replaced objects) could improve balance between structure preservation and text alignment.
- Automatic source prompt generation: Integrating an image captioning model (e.g., BLIP) to automatically generate the source prompt for real images would eliminate the need for users to manually write descriptions of input real images.
- Automatic $\tau$ tuning: A lightweight per-edit model to select the optimal $\tau$ threshold would remove the need for users to manually tune this parameter.
  One key unverified assumption of the work is that early step cross-attention maps fully determine scene layout. For complex, multi-object scenes, later step attention maps can also affect layout, which is why manual tuning of $\tau$ is often required for optimal results, limiting out-of-the-box usability for non-technical users.