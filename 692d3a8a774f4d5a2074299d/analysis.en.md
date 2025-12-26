# 1. Bibliographic Information
## 1.1. Title
Phenaki: Variable Length Video Generation From Open Domain Textual Description

The title clearly states the paper's core contributions: a model named "Phenaki" that can generate videos of varying lengths based on free-form text descriptions.

## 1.2. Authors
Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Dumitru Erhan (all from Google Brain), Santiago Castro (University of Michigan), and Julius Kunze (University College London).

The author list is predominantly from Google Brain, a leading industrial research lab in artificial intelligence. This indicates that the research is well-resourced and positioned at the forefront of large-scale generative modeling.

## 1.3. Journal/Conference
The paper was published on arXiv, a preprint server. This means it is a preliminary version of the research shared with the scientific community before or during peer review for a formal conference or journal. arXiv is a standard platform for rapid dissemination of results in fields like machine learning.

## 1.4. Publication Year
The paper was submitted to arXiv on October 5, 2022.

## 1.5. Abstract
The abstract introduces Phenaki, a model for generating realistic videos from a sequence of text prompts. It identifies key challenges in text-to-video synthesis: high computational cost, a shortage of high-quality text-video data, and the need to handle videos of variable lengths. To overcome these, the authors propose a novel video tokenizer that compresses videos into discrete tokens using causal attention over time, enabling it to work with variable-length inputs. Video generation is handled by a bidirectional masked transformer that converts text tokens into video tokens. The authors highlight a key strategy: jointly training the model on a massive corpus of image-text pairs and a smaller set of video-text pairs, which allows the model to generalize concepts learned from images to video generation. The paper claims two major advancements: Phenaki is the first model to generate arbitrarily long videos from a time-varying sequence of prompts (a "story"), and its video encoder-decoder is more efficient and produces better spatio-temporal consistency than per-frame baselines.

## 1.6. Original Source Link
- **Original Source Link:** https://arxiv.org/abs/2210.02399
- **PDF Link:** https://arxiv.org/pdf/2210.02399v1.pdf
- **Publication Status:** This is a preprint and has not been published in a peer-reviewed journal or conference at the time of this analysis.

# 2. Executive Summary
## 2.1. Background & Motivation
- **Core Problem:** The paper addresses the complex task of generating high-quality, temporally coherent videos of variable lengths from open-domain textual descriptions. A significant extension of this problem is generating a single continuous video that evolves according to a sequence of prompts, akin to a visual story.

- **Existing Challenges:**
    1.  **Data Scarcity:** While massive text-image datasets with billions of examples exist (e.g., LAION-5B), high-quality text-video datasets are significantly smaller (e.g., WebVid with ~10 million videos). This data gap limits the conceptual diversity and quality of video generation models.
    2.  **Computational Cost:** Videos are sequences of images, making them incredibly high-dimensional. Training generative models on raw video data is computationally prohibitive, and even generating tokens for long videos can overwhelm existing transformer models.
    3.  **Variable Length and Temporal Coherence:** Previous models often treated videos as a sequence of independent frames or were restricted to generating fixed-length clips. This leads to poor temporal consistency (flickering, disjointed motion) and lacks the flexibility to generate videos of arbitrary length.
    4.  **Single-Prompt Limitation:** Most text-to-video models are designed to interpret a single, static prompt for an entire clip. They lack the capability to dynamically alter the video's content based on a changing narrative or a sequence of prompts.

- **Paper's Innovative Idea:** The authors' central idea is to tackle these challenges with a new, highly efficient video representation and a flexible generation process. They propose **`C-ViViT`**, a causal video tokenizer that compresses video frames while respecting the arrow of time, allowing it to handle variable lengths naturally. They then introduce a training paradigm that **leverages the vast knowledge in image-text datasets** to enrich the video generation process. Finally, they design an auto-regressive generation scheme at the clip level, enabling the model to produce long videos that follow a "story" composed of multiple, sequential prompts.

## 2.2. Main Contributions / Findings
The paper makes several key contributions to the field of generative AI:

1.  **Phenaki Model:** A novel text-to-video generation system capable of producing diverse, temporally coherent videos of variable lengths (up to several minutes) from open-domain text.

2.  **Time-Variable Text Conditioning (Storytelling):** To the best of the authors' knowledge, this is the **first work to demonstrate video generation from a sequence of changing text prompts**. This allows for creating dynamic visual narratives where the scene evolves according to a story. The example in Figure 1, where a teddy bear seamlessly morphs into a panda as the prompt changes, showcases this unique capability.

3.  **Causal Video Tokenizer (`C-ViViT`):** A new video encoder-decoder architecture that is causal in the time dimension. Unlike previous methods, it processes videos auto-regressively, making it inherently suitable for variable-length inputs and outputs. It also achieves better compression and superior spatio-temporal consistency compared to per-frame tokenization methods.

4.  **Effective Joint Training on Image and Video Data:** The paper demonstrates a successful strategy for training on a combination of large image-text datasets and smaller video-text datasets. This allows the model to learn a wide range of visual concepts and styles from static images (e.g., "pencil drawing") and apply them to dynamic video generation, overcoming the data limitations of video-only training.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
- **Transformer:** A neural network architecture that relies on a mechanism called **`self-attention`** to weigh the importance of different parts of the input sequence. Unlike recurrent neural networks (RNNs) that process data sequentially, transformers can process the entire sequence at once, making them highly parallelizable and effective for modeling long-range dependencies. The core formula for `self-attention` is:
  \$
  \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  \$
  -   $Q$ (Query), $K$ (Key), and $V$ (Value) are matrices derived from the input embeddings. The attention mechanism computes a score between each query and all keys, scales it, applies a `softmax` to get weights, and then uses these weights to compute a weighted sum of the values. This allows a model to focus on the most relevant parts of the input when producing an output.

- **Vector Quantization (VQ):** A technique used to compress data by mapping continuous or high-dimensional vectors to a finite set of "codewords" in a dictionary or "codebook." In generative models like VQ-VAE or VQ-GAN, an encoder maps an image or video patch to a continuous vector, which is then replaced by the nearest vector from a learned codebook. This process discretizes the latent space, turning the visual data into a sequence of discrete "tokens," similar to words in a sentence. This tokenization is crucial for using powerful sequence models like transformers for generation.

- **Auto-regressive Models:** These are generative models that produce a sequence of data one step at a time, where each new output is conditioned on all the previously generated outputs. For example, when generating a sentence, the model predicts the next word based on the words it has already written. This sequential nature guarantees a coherent structure but can be slow during inference. Phenaki's `C-ViViT` is auto-regressive in time, meaning each frame's representation depends on the previous frames.

- **Bidirectional Masked Transformers:** Inspired by models like BERT, these transformers are trained to predict missing (masked) parts of a sequence using context from both before and after the mask. During inference, this allows for a non-auto-regressive, parallel decoding process. `MaskGIT`, used by Phenaki, starts with a fully masked sequence of tokens and iteratively fills them in over a fixed number of steps, making generation much faster than traditional auto-regressive models.

- **Classifier-Free Guidance:** A technique to improve the adherence of a generated sample to its conditioning signal (e.g., a text prompt). During training, the model is randomly trained with and without the text prompt. At inference, the model's output is guided by taking a weighted combination of the conditional and unconditional predictions. This "guides" the generation towards samples that are more strongly aligned with the prompt, often at the cost of some diversity.

## 3.2. Previous Works
- **Text-to-Image Models:** The paper builds upon successes in text-to-image generation.
    -   `DALL-E` and `Parti` are auto-regressive models that first tokenize an image using a VQ-VAE or VQ-GAN and then use a large transformer to generate these image tokens from text tokens.
    -   `Imagen` is a diffusion model that uses a frozen text encoder to guide an image generation process that starts from noise and gradually refines it.
    -   Phenaki adopts the token-based approach from models like `Parti` but adapts it for the video domain.

- **Text-to-Video Models:**
    -   `GODIVA`, `NUWA`, and `CogVideo` are prior transformer-based text-to-video models. The paper argues their main limitation is treating videos as a sequence of independent images, using per-frame image encoders (`VQ-GAN`). This approach struggles to model motion and temporal dynamics effectively and results in a very large number of tokens, making it computationally expensive for long videos.
    -   `NUWA-Infinity` tried to address variable-length generation but used a second layer of auto-regression, which can be complex.
    -   `Video Diffusion Models (VDM)` can generate high-quality videos but are typically limited to fixed lengths and have very slow sampling times, making long video generation impractical.

- **Video Representation Models:**
    -   `ViViT` (Video Vision Transformer) was a key inspiration for `C-ViViT`. However, the original `ViViT` used all-to-all attention in the time dimension, meaning every frame could see every other frame. This makes it inherently unsuitable for variable-length inputs or auto-regressive generation into the future.
    -   `VideoVQVAE` is a fixed-length video encoder, which is also not suitable for the goals of Phenaki.

## 3.3. Technological Evolution
The field of visual generation has evolved from generating static images to dynamic videos. Early work focused on GANs and VAEs for images. The advent of transformers and large-scale pre-training led to powerful text-to-image models. The next logical step was text-to-video. Early attempts often repurposed image models, leading to videos that looked like "moving images" with poor coherence. Recent work, including Phenaki, focuses on creating architectures that are fundamentally designed for video, treating space and time as intertwined dimensions. Phenaki sits at a point in this evolution where the focus is shifting from generating short, fixed-size clips to creating long, controllable, and narratively complex videos.

## 3.4. Differentiation Analysis
Phenaki's core innovations differentiate it from prior work in several ways:
- **Architecture (`C-ViViT` vs. Per-Frame Encoders):** While models like `CogVideo` and `NUWA` use a standard image VQ-GAN on each frame independently, Phenaki's `C-ViViT` uses spatio-temporal patches and **causal temporal attention**. This allows it to explicitly model motion and dependencies between frames, resulting in better temporal coherence and a more compressed token representation.
- **Flexibility (Variable vs. Fixed Length):** The causal nature of `C-ViViT` and the clip-level auto-regressive generation process allow Phenaki to generate videos of arbitrary length. This is a significant advantage over models like `GODIVA` or the base `VDM`, which are designed for fixed-size outputs.
- **Conditioning (Story vs. Single Prompt):** Phenaki is the first model designed to handle a sequence of time-varying prompts. This moves beyond simple "text-to-clip" generation and opens the door to "text-to-story" generation, a much more expressive and creative task.
- **Training Data (Joint vs. Video-Only):** Phenaki explicitly formulates a strategy to combine large image-text datasets with smaller video-text datasets. This allows it to learn a vast vocabulary of visual concepts from images and transfer that knowledge to video, a crucial advantage given the relative scarcity of video data.

# 4. Methodology
The Phenaki model consists of three main components: a pre-trained text encoder, a novel video encoder-decoder (`C-ViViT`), and a bidirectional transformer for generation (`MaskGIT`).

The overall architecture is shown in Figure 2 from the original paper:

![Figure 2. The architecture of Phenaki. Left: C-ViViT encoder architecture. The embeddings of images and video patches from raw frames x are processed by a spatial and then a causal transformer (auto-regressive in time) to generate video tokens z. Center: MaskGiT is trained to reconstruct masked tokens $\\mathbf { z }$ predicted by a frozen C-ViViT encoder and conditioned on T5X tokens of a given prompt $\\mathbf { p } _ { 0 }$ . Right: How Phenaki can generate arbitrary long videos by freezing the past token and generating the future tokens. The prompt can change over time to enable time-variable prompt (i.e. story) conditional generation. The subscripts represent time (i.e. frame number).](images/2.jpg)
*该图像是Phenaki架构的示意图，展示了C-ViViT编码器、训练变换器和视频生成模块的结构。其中，编码器通过空间和因果变换器生成视频令牌 $z$，训练变换器利用随机掩码重建掩码令牌，而视频生成模块则通过固定过去令牌生成未来令牌，支持基于时间变化的提示。图中包含了多个操作和令牌的状态。*

## 4.1. C-ViViT: The Encoder-Decoder Video Model
The first major contribution is `C-ViViT`, a model designed to learn a compressed, discrete representation (tokens) of a video. Its key feature is being **auto-regressive in time**, which is what enables variable-length video processing.

### 4.1.1. Encoder Architecture
The goal of the encoder is to take a video sequence $\mathbf{x} \in \mathbb{R}^{(t_x+1) \times h_x \times w_x \times c_x}$ and compress it into a sequence of discrete tokens. The process is as follows:

1.  **Asymmetric Patching:** The model treats the first frame differently from the rest.
    *   The **first frame** is divided into non-overlapping 2D image patches.
    *   The **subsequent frames** are grouped and divided into non-overlapping 3D spatio-temporal "tubelet" patches (e.g., $2 \times 8 \times 8$ pixels in time, height, width).
        This design choice is crucial because it allows an image to be processed just like the first frame of a video, enabling seamless joint training on both image and video datasets.

2.  **Linear Projection:** Each patch is flattened into a vector and linearly projected into a $d_z$-dimensional embedding space.

3.  **Spatial Transformer:** A standard transformer with all-to-all `self-attention` is applied across the spatial dimensions for each time step. This allows the model to capture spatial relationships within each frame (or group of frames represented by the tubelets).

4.  **Temporal Transformer:** This is the core innovation. A transformer with **causal attention** is applied across the temporal dimension. This means that when computing the representation for a token at time $t$, the model can only attend to tokens from time steps less than or equal to $t$. This enforces a strict temporal order and makes the model auto-regressive, allowing it to encode videos of any length and extrapolate into the future.

5.  **Vector Quantization (VQ):** The continuous output embeddings $\mathbf{z}$ from the temporal transformer are quantized. Each embedding is replaced by the closest vector from a learned codebook $\mathbf{E}$. This produces the final sequence of discrete video tokens.

### 4.1.2. Quantization and Losses
To train `C-ViViT`, the paper uses a combination of losses to ensure high-quality reconstruction.

-   **VQ Loss:** The standard VQ-VAE loss is used to train the codebook and encourage the encoder outputs to be close to the codebook entries. The formula is:
    \$
    L_{VQ} = \lVert \mathbf{sg(z)} - \mathbf{e} \rVert_2^2 + \beta \lVert \mathbf{z} - sg(\mathbf{e}) \rVert_2^2
    \$
    -   $\mathbf{z}$ is the continuous output of the encoder.
    -   $\mathbf{e}$ is the closest codebook vector to $\mathbf{z}$.
    -   $\mathrm{sg}(\cdot)$ is the stop-gradient operator, which prevents gradients from flowing through its argument. The first term updates the codebook, and the second term (the commitment loss, weighted by $\beta$) encourages the encoder to produce outputs that commit to a codebook vector.

-   **Total Loss:** The full training objective for `C-ViViT` is a weighted sum of multiple loss terms designed to improve perceptual quality:
    \$
    L = L_{VQ} + 0.1 \times L_{Adv} + 0.1 \times L_{IP} + 1.0 \times L_{VP} + 1.0 \times L_2
    \$
    -   $L_{Adv}$: An adversarial loss (using a StyleGAN discriminator) to make the reconstructed videos look more realistic.
    -   $L_{IP}$: An image perceptual loss, which measures distance in the feature space of a pre-trained network (like VGG or I3D) to better match human perception of image quality.
    -   $L_{VP}$: A video perceptual loss, using features from the I3D network to ensure the temporal dynamics are well-reconstructed.
    -   $L_2$: A standard pixel-wise reconstruction loss (mean squared error).

### 4.1.3. Decoder Architecture
The decoder is simply the inverse of the encoder. It takes the quantized embeddings, passes them through a temporal transformer, then a spatial transformer, and finally a linear projection layer to reconstruct the video pixels.

## 4.2. Text-to-Video Generation with Bidirectional Transformers
Once `C-ViViT` is trained, it is frozen and used as a video tokenizer. The second stage is to train a transformer model to generate these video tokens from text.

### 4.2.1. Masked Bidirectional Transformer (`MaskGIT`)
Instead of an auto-regressive transformer that generates tokens one by one (which is slow), Phenaki uses a bidirectional transformer based on `MaskGIT`. This model is trained on a "mask and predict" task.

-   **Training:**
    1.  A video is converted to a sequence of tokens $\mathbf{a}$ by the `C-ViViT` encoder.
    2.  A random portion of these tokens is replaced with a special `[MASK]` token.
    3.  The `MaskGIT` model is trained to predict the original identity of the masked tokens, conditioned on the unmasked tokens and a text embedding $\mathbf{p}$ (obtained from a pre-trained T5X model).
        The loss function is the cross-entropy loss over the masked tokens:
    \$
    L_{\mathrm{mask}} = - \sum_{\forall i \in [1, N], m_i = 1} \log p(a_i | \mathbf{a}_{\bar{M}}, \mathbf{p})
    \$
    -   $m_i=1$ indicates that token $a_i$ is masked.
    -   $\mathbf{a}_{\bar{M}}$ represents the set of unmasked tokens.
    -   $\mathbf{p}$ is the text conditioning.

-   **Inference:**
    Generation is an iterative process that is much faster than auto-regressive decoding.
    1.  Start with a sequence of all `[MASK]` tokens.
    2.  In each step, the model predicts all masked tokens in parallel.
    3.  The most confident predictions are kept (unmasked), and the rest are re-masked.
    4.  This process is repeated for a small, fixed number of steps (e.g., 12 to 48), gradually revealing the full video token sequence.

### 4.2.2. Auto-regressive Generation of Long Videos (Storytelling)
This is the mechanism that enables generating long videos and visual stories. It works auto-regressively at the **clip level**.

1.  **Generate First Clip:** Given the first prompt, `MaskGIT` generates the tokens for the first video clip. These tokens are decoded by `C-ViViT` into video frames.

2.  **Extrapolate:** To generate the next clip:
    *   Take the last few frames (e.g., $K=5$) of the just-generated clip.
    *   Encode these frames using the `C-ViViT` encoder to get their corresponding video tokens. These tokens are now considered "known" or "past context."
    *   Initialize the `MaskGIT` inference process with these known tokens at the beginning of the sequence and `[MASK]` tokens for the rest.
    *   Provide the next text prompt (which can be the same or different).
    *   Run the iterative `MaskGIT` sampling to generate the tokens for the new frames.

3.  **Repeat:** This process can be repeated indefinitely to generate arbitrarily long videos, with the prompt changing at the start of each new clip to tell a story. This is elegantly illustrated in the rightmost panel of Figure 2.

# 5. Experimental Setup
## 5.1. Datasets
-   **Training Data:** The model was trained on a large, mixed corpus of internal and public datasets.
    -   **Video-Text Data:** An internal dataset of `~15 million` text-video pairs, captured at 8 FPS.
    -   **Image-Text Data:** A combination of an internal dataset with `~50 million` text-image pairs and the public **LAION-400M** dataset (`~400 million` pairs). The paper explores different mixing ratios, with a typical setup being 80% video data and 20% image data.

-   **Evaluation Datasets:**
    -   **Kinetics-400 & Kinetics-600:** Large-scale, high-quality datasets of human action videos from YouTube. Used for zero-shot text-to-video evaluation and video prediction.
    -   **Moments-in-Time (MiT):** A large dataset with over 1 million labeled videos focusing on events and actions. Used to evaluate the reconstruction quality of `C-ViViT`.
    -   **BAIR Robot Pushing:** A standard benchmark dataset for video prediction, containing videos of a robot arm pushing objects.

## 5.2. Evaluation Metrics
-   **FID (Fréchet Inception Distance):**
    1.  **Conceptual Definition:** FID measures the similarity between two distributions of images, typically real images and generated images. It calculates the distance between the distributions of features extracted from a pre-trained InceptionV3 network. A lower FID score indicates that the generated images are more similar to the real images in terms of feature representation, suggesting higher quality and diversity.
    2.  **Mathematical Formula:**
        \$
        \mathrm{FID}(x, g) = ||\mu_x - \mu_g||^2_2 + \mathrm{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x\Sigma_g)^{1/2})
        \$
    3.  **Symbol Explanation:**
        -   $\mu_x$ and $\mu_g$ are the mean vectors of the Inception features for the real and generated image distributions, respectively.
        -   $\Sigma_x$ and $\Sigma_g$ are the covariance matrices of the features.
        -   $\mathrm{Tr}(\cdot)$ denotes the trace of a matrix.

-   **FVD (Fréchet Video Distance):**
    1.  **Conceptual Definition:** FVD is the video-domain equivalent of FID. It measures the quality and temporal coherence of generated videos by comparing their feature distributions to those of real videos. The features are extracted from a pre-trained video classification model (I3D), which is sensitive to both appearance and motion. A lower FVD score indicates better video quality and more realistic dynamics.
    2.  **Mathematical Formula:** The formula is identical in structure to FID, but the features are extracted from an I3D network instead of InceptionV3.
    3.  **Symbol Explanation:** Same as FID, but $\mu$ and $\Sigma$ now refer to the statistics of I3D video features.

-   **CLIP Score:**
    1.  **Conceptual Definition:** This metric evaluates how well a generated image or video aligns with its text prompt. It uses the pre-trained CLIP (Contrastive Language-Image Pre-Training) model, which can embed both images and text into a shared latent space. The CLIP score is the cosine similarity between the CLIP embedding of the generated visual content and the CLIP embedding of the text prompt. A higher score means better semantic alignment.
    2.  **Mathematical Formula:**
        \$
        \text{CLIP Score} = \cos(\mathbf{v}, \mathbf{t}) = \frac{\mathbf{v} \cdot \mathbf{t}}{||\mathbf{v}|| \cdot ||\mathbf{t}||}
        \$
    3.  **Symbol Explanation:**
        -   $\mathbf{v}$ is the feature vector of the generated image/video from the CLIP visual encoder.
        -   $\mathbf{t}$ is the feature vector of the text prompt from the CLIP text encoder.

## 5.3. Baselines
-   **For Text-to-Video:** `T2V`, `SC TFGAN`, `NUWA`. These represent earlier methods in text-to-video synthesis.
-   **For Video Reconstruction:** `Convolutional VQ-GAN` and `ViT VQ-GAN`. These are per-frame image-based tokenizers, serving as a direct comparison to show the benefits of `C-ViViT`'s spatio-temporal approach.
-   **For Video Prediction:** A wide range of state-of-the-art models including `Video Transformer`, `DVD-GAN`, `Transframer`, `Video Diffusion`, and `CogVideo`.

# 6. Results & Analysis
## 6.1. Core Results Analysis
Phenaki's effectiveness is demonstrated across several tasks, with both quantitative and qualitative evidence.

## 6.1.1. Text-Conditional Video Generation
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>FID Image →</th>
<th>FID Video V</th>
</tr>
</thead>
<tbody>
<tr>
<td>T2V [25]</td>
<td>82.13</td>
<td>33.51</td>
</tr>
<tr>
<td>SC TFGAN [5]</td>
<td>14.65</td>
<td>7.34</td>
</tr>
<tr>
<td>NUWA</td>
<td>31.76</td>
<td>7.19</td>
</tr>
<tr>
<td>Phenaki [0-Shot]</td>
<td>28.46</td>
<td>7.05</td>
</tr>
</tbody>
</table>

**Analysis:** On the Kinetics-400 benchmark, Phenaki is evaluated in a **zero-shot** setting (it was not trained or fine-tuned on Kinetics). Despite this, it achieves a competitive `FID` score and a state-of-the-art `FVD` score compared to previous methods that were trained on this dataset. The significantly low `FVD` score (lower is better) is particularly important, as it confirms that Phenaki's architecture excels at generating videos with realistic and coherent motion.

Qualitative examples, like those in Figure 3, show the model's ability to generate videos for complex, compositional prompts like "a panda bear swimming, pencil drawing" or "an astronaut riding a horse in space."

![Figure 3. Text conditional video generation. Each row shows selected frames from a video generated given the prompt. The model is trained on a mix of images and videos. The video dataset does not include any stylized videos such as pencil drawings, however, the image dataset does. The model can generalize from still images to videos. This figure also demonstrate the capability of the model in generating new unseen compositions. Full videos are available at phenaki.github.io.](images/3.jpg)
*该图像是插图，展示了基于文本生成的视频示例。每一行显示了根据给定提示生成的不同场景，内容包括可爱的熊猫、宇航员等，展示了模型在多种情境下的创造能力。*

## 6.1.2. Importance of Joint Text-to-Image and Text-to-Video Training
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Data Split<br>Vid% / Img%</th>
<th colspan="2">Text to Video</th>
<th colspan="2">Text to Image</th>
</tr>
<tr>
<th>CLIP ↑</th>
<th>FID ↓</th>
<th>FVD ↓</th>
<th>CLIP ↑</th>
<th>FID ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>100% / 0%</td>
<td>0.298</td>
<td>19.2</td>
<td>168.9</td>
<td>0.240</td>
<td>53.9</td>
</tr>
<tr>
<td>80% / 20%</td>
<td>0.303</td>
<td>21.4</td>
<td>198.4</td>
<td>0.289</td>
<td>29.4</td>
</tr>
<tr>
<td>50% / 50%</td>
<td>0.302</td>
<td>21.4</td>
<td>239.7</td>
<td>0.287</td>
<td>30.5</td>
</tr>
</tbody>
</table>

**Analysis:** This table reveals a crucial trade-off.
-   Training on **100% video data** yields the best `FVD` score (168.9), indicating the highest quality video dynamics. However, its text-image alignment (`CLIP`) and image generation quality (`FID`) are the worst.
-   As more **image data is added** (moving from 0% to 50%), the text alignment (`CLIP` score) and image quality (`FID` on LAION) improve dramatically. This is because the image datasets are much larger and contain a wider variety of concepts.
-   However, this improvement comes at the cost of a worse `FVD`, suggesting that learning from static images slightly degrades the model's ability to generate fluid motion.
    This experiment validates the authors' hypothesis that joint training is essential for learning diverse concepts, even if it introduces a trade-off with motion quality.

## 6.1.3. Visual Storytelling and Long Video Generation
The most novel capability of Phenaki is its ability to generate long videos from a sequence of prompts. Figure 1 and Figure 5 show compelling examples. In Figure 1, the model transitions from "A photorealistic teddy bear is swimming" to "A panda bear is swimming," smoothly morphing the subject while maintaining the background context. This demonstrates true temporal understanding and control, going far beyond what single-prompt models can achieve.

![Figure 1. Time variable text (i.e. story) conditional video generation. The entire figure is one continuous video generated auto-regressively. We start by generating the video conditioned on the first prompt and then after a couple of frames we change the prompt to the next one. Each row contains a selected number of frames (from left to right in order) while the model was conditioned on that particular prompt. The model manages to preserve the temporal coherence of the video while adapting to the new prompt, usually taking the shortest path for the adaption (notice the morphing of the teddy bear to the panda). Please note that the generated video has complex visual features such as reflections, occlusions, interactions and scene transitions. Full video is available at phenaki.github.io.](images/1.jpg)
*该图像是示意图，展示了一个变换的故事情节，其中玩具熊在水中游动，随着不同提示生成了不同的画面。第一行为提示"玩具熊在水下"，逐渐转变为"熊猫在水中游泳"，展现了平滑的视觉过渡和时序一致性。*

## 6.1.4. Video Encoding (`C-ViViT` Performance)
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>FID ↓</th>
<th>FVD ↓</th>
<th>Number of Tokens ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Conv VQ-GAN [12]</td>
<td>7.5</td>
<td>306.1</td>
<td>2560</td>
</tr>
<tr>
<td>Conv VQ-GAN + Video loss</td>
<td>13.7</td>
<td>346.5</td>
<td>2560</td>
</tr>
<tr>
<td>ViT VQ-GAN [58]</td>
<td>3.4</td>
<td>166.6</td>
<td>2560</td>
</tr>
<tr>
<td>ViT VQ-GAN + Video loss</td>
<td>3.8</td>
<td>173.1</td>
<td>2560</td>
</tr>
<tr>
<td>C-ViViT VQ-GAN (Ours)</td>
<td>4.5</td>
<td>65.78</td>
<td>1536</td>
</tr>
</tbody>
</table>

**Analysis:** This table compares `C-ViViT` with per-frame image tokenizers on the task of video reconstruction.
-   **Temporal Coherence (`FVD`):** `C-ViViT` achieves a massively better `FVD` score (65.78) compared to the best per-frame baseline (166.6). This is direct evidence that its spatio-temporal architecture is far superior at modeling and reconstructing video dynamics.
-   **Efficiency (`Number of Tokens`):** `C-ViViT` compresses an 11-frame video into only 1536 tokens, a 40% reduction compared to the 2560 tokens required by per-frame methods for 10 frames. This makes the downstream generation task with the `MaskGIT` transformer significantly less computationally expensive.
-   **Per-frame Quality (`FID`):** While `C-ViViT`'s `FID` is slightly worse than the ViT VQ-GAN, the huge gain in `FVD` and token efficiency makes it a clear winner for the video domain.

## 6.1.5. Video Prediction
The following are the results from Table 4 (Kinetics-600) and Table 5 (BAIR) of the original paper:

<table><caption>Table 4. Video prediction on Kinetics-600 [7]. While Phenaki is not designed for video prediction it achieves comparable results with SOTA video prediction models.</caption>
<thead>
<tr>
<th>Method</th>
<th>FVD ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>Video Transformer [51]</td>
<td>170.0 ± 5.00</td>
</tr>
<tr>
<td>CogVideo [18]</td>
<td>109.2</td>
</tr>
<tr>
<td>DVD-GAN-FP [9]</td>
<td>69.1 ± 0.78</td>
</tr>
<tr>
<td>Video VQ-VAE [49]</td>
<td>64.3 ± 2.04</td>
</tr>
<tr>
<td>CCVS [28]</td>
<td>55.0 ± 1.00</td>
</tr>
<tr>
<td>TrIVD-GAN-FP [27]</td>
<td>25.7 ± 0.66</td>
</tr>
<tr>
<td>Transframer [31]</td>
<td>25.4</td>
</tr>
<tr>
<td>RaMViD [19]</td>
<td>16.5</td>
</tr>
<tr>
<td>Video Diffusion [17]</td>
<td>16.2 ± 0.34</td>
</tr>
<tr>
<td>Phenaki (Ours)</td>
<td>36.4 ± 0.19</td>
</tr>
</tbody>
</table>

<table><caption>Table 5. Video prediction on BAIR [11].</caption>
<tbody>
<tr><td><b>Method</b></td><td><b>FVD↓</b></td></tr>
<tr><td>DVD-GAN [9]</td><td>109.8</td></tr>
<tr><td>VideoGPT [55]</td><td>103.3</td></tr>
<tr><td>TrIVD-GAN [27]</td><td>103.3</td></tr>
<tr><td>Transframer [31]</td><td>100.0</td></tr>
<tr><td>HARP [57]</td><td>99.3</td></tr>
<tr><td>CCVS [28]</td><td>99.0</td></tr>
<tr><td>Video Transformer [51]</td><td>94.0</td></tr>
<tr><td>FitVid [3]</td><td>93.6</td></tr>
<tr><td>MCVD [47]</td><td>89.5</td></tr>
<tr><td>NUWA [54]</td><td>86.9</td></tr>
<tr><td>RaMViD [19]</td><td>84.2</td></tr>
<tr><td>Phenaki (Ours)</td><td>97.0</td></tr>
</tbody>
</table>

**Analysis:** Even though Phenaki was not specifically designed for the standard video prediction task (generating future frames given past frames), it performs competitively against state-of-the-art specialized models on both Kinetics-600 and BAIR. This result serves as further validation that the video representations learned by `C-ViViT` are powerful and capture meaningful dynamics.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
The paper introduces Phenaki, a groundbreaking text-to-video model that makes significant strides in generating long, variable-length videos from open-domain text. Its core contributions are twofold: the novel **`C-ViViT`** architecture, a causal video tokenizer that efficiently compresses video while preserving temporal coherence, and the demonstration of **story-based video generation** from a sequence of time-varying prompts. The authors also show that jointly training on massive image datasets and smaller video datasets is a highly effective strategy for improving the conceptual range of video generation. The model achieves strong performance across multiple benchmarks, including text-to-video synthesis and video prediction, establishing a new state of the art for flexible and controllable video generation.

## 7.2. Limitations & Future Work
-   **Data-Driven Trade-offs:** The authors acknowledge the trade-off between conceptual diversity (gained from image data) and motion quality (`FVD`). As larger and more diverse high-quality video-text datasets become available, this trade-off may be mitigated.
-   **Video Quality:** While a significant step forward, the generated videos are not yet consistently photorealistic and can sometimes contain artifacts, especially during complex transitions between prompts. The authors note that the quality is not yet "indistinguishable from real videos."
-   **Ethical Concerns:** In the Ethics Statement, the authors discuss the potential for misuse, such as generating malicious fake content (deepfakes) and spreading misinformation. They also acknowledge that the model is trained on web-scale datasets like LAION-400M, which are known to contain biases, violence, and pornography. Due to these concerns, they made the responsible decision not to release the model, code, or demo publicly at the time of publication.

## 7.3. Personal Insights & Critique
-   **Architectural Elegance:** The design of `C-ViViT` is particularly insightful. Introducing causality in the time dimension is an elegant and effective solution to the long-standing problem of handling variable-length videos. The asymmetric patching to seamlessly integrate image and video data is also a clever design choice that directly addresses the data scarcity problem.
-   **Paradigm Shift in Generation:** The concept of "storytelling" with time-varying prompts is a significant paradigm shift. It moves the field beyond generating isolated clips towards creating dynamic, narrative-driven content. This opens up immense possibilities for creative applications in art, entertainment, and design, potentially serving as a powerful tool for visual artists and filmmakers.
-   **The Next Frontier:** Phenaki's architecture is transformer-based. A potential future direction would be to combine its powerful video representation (`C-ViViT`) with the high-fidelity generation capabilities of diffusion models. A diffusion model operating in the latent space defined by `C-ViViT` could potentially achieve even higher visual quality while retaining the flexibility of variable-length generation.
-   **Unverified Assumptions:** While the results are impressive, the "morphing" effect between prompts might not always be the desired behavior. For stories with distinct scene changes, a hard cut might be more appropriate than a seamless transition. Future work could explore giving users more explicit control over the type of transition between prompts.
-   **Critique:** The paper relies heavily on private, internal datasets, which makes direct reproduction of the results difficult for the broader research community. While understandable for an industrial lab, this is a common challenge in large-scale generative modeling research. Nonetheless, the architectural principles and findings are clearly articulated and provide valuable insights for the field.