# Taming Transformers for High-Resolution Image Synthesis

Patrick Esser\* Robin Rombach\* Björn Ommer Heidelberg Collaboratory for Image Processing, IWR, Heidelberg University, Germany \*Both authors contributed equally to this work

![](images/1.jpg)

# Abstract

Designed to learn long-range interactions on sequential data, transformers continue to show state-of-the-art results on a wide variety of tasks. In contrast to CNNs, they contain no inductive bias that prioritizes local interactions. This makes them expressive, but also computationally infeasible for long sequences, such as high-resolution images. We demonstrate how combining the effectiveness of the inductive bias of CNNs with the expressivity of transformers enables them to model and thereby synthesize high-resolution images. We show how to (i) use CNNs to learn a contextrich vocabulary of image constituents, and in turn (ii) utilize transformers to efficiently model their composition within high-resolution images. Our approach is readily applied to conditional synthesis tasks, where both non-spatial information, such as object classes, and spatial information, such as segmentations, can control the generated image. In particular, we present the first results on semanticallyguided synthesis of megapixel images with transformers and obtain the state of the art among autoregressive models on class-conditional ImageNet. Code and pretrained models can be found at https://git.io/JnyvK.

# 1. Introduction

Transformers are on the rise—they are now the de-facto standard architecture for language tasks [74, 57, 58, 5] and are increasingly adapted in other areas such as audio [12] and vision [8, 16]. In contrast to the predominant vision architecture, convolutional neural networks (CNNs), the transformer architecture contains no built-in inductive prior on the locality of interactions and is therefore free to learn complex relationships among its inputs. However, this generality also implies that it has to learn all relationships, whereas CNNs have been designed to exploit prior knowledge about strong local correlations within images. Thus, the increased expressivity of transformers comes with quadratically increasing computational costs, because all pairwise interactions are taken into account. The resulting energy and time requirements of state-of-the-art transformer models thus pose fundamental problems for scaling them to high-resolution images with millions of pixels.

Observations that transformers tend to learn convolutional structures [16] thus beg the question: Do we have to re-learn everything we know about the local structure and regularity of images from scratch each time we train a vision model, or can we efficiently encode inductive image biases while still retaining the flexibility of transformers? We hypothesize that low-level image structure is well described by a local connectivity, i.e. a convolutional architecture, whereas this structural assumption ceases to be effective on higher semantic levels. Moreover, CNNs not only exhibit a strong locality bias, but also a bias towards spatial invariance through the use of shared weights across all positions. This makes them ineffective if a more holistic understanding of the input is required.

Our key insight to obtain an effective and expressive model is that, taken together, convolutional and transformer architectures can model the compositional nature of our visual world [51]: We use a convolutional approach to efficiently learn a codebook of context-rich visual parts and, subsequently, learn a model of their global compositions. The long-range interactions within these compositions require an expressive transformer architecture to model distributions over their consituent visual parts. Furthermore, we utilize an adversarial approach to ensure that the dictionary of local parts captures perceptually important local structure to alleviate the need for modeling low-level statistics with the transformer architecture. Allowing transformers to concentrate on their unique strength—modeling long-range relations—enables them to generate high-resolution images as in Fig. 1, a feat which previously has been out of reach. Our formulationgives control over the generated images by means of conditioning information regarding desired object classes or spatial layouts. Finally, experiments demonstrate that our approach retains the advantages of transformers by outperforming previous codebook-based state-of-the-art approaches based on convolutional architectures.

# 2. Related Work

The Transformer Family The defining characteristic of the transformer architecture [74] is that it models interactions between its inputs solely through attention [2, 36, 52] which enables them to faithfully handle interactions between inputs regardless of their relative position to one another. Originally applied to language tasks, inputs to the transformer were given by tokens, but other signals, such as those obtained from audio [41] or images [8], can be used. Each layer of the transformer then consists of an attention mechanism, which allows for interaction between inputs at different positions, followed by a position-wise fully connected network, which is applied to all positions independently. More specifically, the (self-)attention mechanism can be described by mapping an intermediate representation with three position-wise linear layers into three representations, query $Q \in \mathbb { R } ^ { N \times d _ { k } }$ , key $\dot { K } \in \mathbb { R } ^ { N \times d _ { k } }$ and value $V \in \mathbb { R } ^ { N \times d _ { v } }$ , to compute the output as

$$
\operatorname { A t t n } ( Q , K , V ) = \mathrm { s o f t m a x } \Big ( \frac { Q K ^ { t } } { \sqrt { d _ { k } } } \Big ) V \in \mathbb { R } ^ { N \times d _ { v } } .
$$

When performing autoregressive maximum-likelihood learning, non-causal entries of $Q K ^ { t }$ , i.e. all entries below its diagonal, are set to $- \infty$ and the final output of the transformer is given after a linear, point-wise transformation to predict logits of the next sequence element. Since the attention mechanism relies on the computation of inner products between all pairs of elements in the sequence, its computational complexity increases quadratically with the sequence length. While the ability to consider interactions between all elements is the reason transformers efficiently learn long-range interactions, it is also the reason transformers quickly become infeasible, especially on images, where the sequence length itself scales quadratically with the resolution. Different approaches have been proposed to reduce the computational requirements to make transformers feasible for longer sequences. [55] and [76] restrict the receptive fields of the attention modules, which reduces the expressivity and, especially for high-resolution images, introduces assumptions on the independence of pixels. [12] and [26] retain the full receptive field but can reduce costs for a sequence of length $n$ only from $n ^ { 2 }$ to $n \sqrt { n }$ , which makes resolutions beyond 64 pixels still prohibitively expensive.

Convolutional Approaches The two-dimensional structure of images suggests that local interactions are particularly important. CNNs exploit this structure by restricting interactions between input variables to a local neighborhood defined by the kernel size of the convolutional kernel. Applying a kernel thus results in costs that scale linearly with the overall sequence length (the number of pixels in the case of images) and quadratically in the kernel size, which, in modern CNN architectures, is often fixed to a small constant such as $3 \times 3$ This inductive bias towards local interactions thus leads to efficient computations, but the wide range of specialized layers which are introduced into CNNs to handle different synthesis tasks [53, 80, 68, 85, 84] suggest that this bias is often too restrictive.

Convolutional architectures have been used for autoregressive modeling of images [70, 71, 10] but, for lowresolution images, previous works [55, 12, 26] demonstrated that transformers consistently outperform their convolutional counterparts. Our approach allows us to efficiently model high-resolution images with transformers while retaining their advantages over state-of-the-art convolutional approaches.

Two-Stage Approaches Closest to ours are two-stage approaches which first learn an encoding of data and afterwards learn, in a second stage, a probabilistic model of this encoding. [13] demonstrated both theoretical and empirical evidence on the advantages of first learning a data representation with a Variational Autoencoder (VAE) [38, 62], and then again learning its distribution with a VAE. [18, 78] demonstrate similar gains when using an unconditional normalizing flow for the second stage, and [63, 64] when using a conditional normalizing flow. To improve training efficiency of Generative Adversarial Networks (GANs), [43] learns a GAN [20] on representations of an autoencoder and [21] on low-resolution wavelet coefficients which are then decoded to images with a learned generator.

![](images/2.jpg)  
Figure 2. Our approach uses a convolutional $V Q G A N$ to learn a codebook of context-rich visual parts, whose composition is subsequently convolutional approaches to transformer based high resolution image synthesis.

[72] presents the Vector Quantised Variational Autoencoder (VQVAE), an approach to learn discrete representations of images, and models their distribution autoregressively with a convolutional architecture. [61] extends this approach to use a hierarchy of learned representations. However, these methods still rely on convolutional density estimation, which makes it difficult to capture long-range interactions in high-resolution images. [8] models images autoregressively with transformers in order to evaluate the suitability of generative pretraining to learn image representations for downstream tasks. Since input resolutions of $3 2 \times 3 2$ pixels are stll quite computationally expensive [8], a VQVAE is used to encode images up to a resolution of $1 9 2 \times 1 9 2$ . In an effort to keep the learned discrete representation as spatially invariant as possible with respect to the pixels, a shallow VQVAE with small receptive field is employed. In contrast, we demonstrate that a powerful first stage, which captures as much context as possible in the learned representation, is critical to enable efficient highresolution image synthesis with transformers.

# 3. Approach

Our goal is to exploit the highly promising learning capabilities of transformer models [74] and introduce them to high-resolution image synthesis up to the megapixel range. Previous work [55, 8] which applied transformers to image generation demonstrated promising results for images up to a size of $6 4 \times 6 4$ pixels but, due to the quadratically increasing cost in sequence length, cannot simply be scaled to higher resolutions.

High-resolution image synthesis requires a model that understands the global composition of images, enabling it to generate locally realistic as well as globally consistent patterns. Therefore, instead of representing an image with pixels, we represent it as a composition of perceptually rich image constituents from a codebook. By learning an effective code, as described in Sec. 3.1, we can significantly reduce the description length of compositions, which allows us to efficiently model their global interrelations within images with a transformer architecture as described in Sec. 3.2. This approach, summarized in Fig. 2, is able to generate realistic and consistent high resolution images both in an unconditional and a conditional setting.

# 3.1. Learning an Effective Codebook of Image Constituents for Use in Transformers

To utilize the highly expressive transformer architecture for image synthesis, we need to express the constituents of an image in the form of a sequence. Instead of building on individual pixels, complexity necessitates an approach that uses a discrete codebook of learned representations, such that any image $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times 3 }$ can be represented by a spatial collection of codebook entries $z _ { \mathbf { q } } \in \mathbb { R } ^ { h \times w \times n _ { z } }$ , where $n _ { z }$ is the dimensionality of codes. An equivalent representation is a sequence of $h \cdot w$ indices which specify the respective entries in the learned codebook. To effectively learn such a discrete spatial codebook, we propose to directly incorporate the inductive biases of CNNs and incorporate ideas from neural discrete representation learning [72]. First, we learn a convolutional model consisting of an encoder $E$ and a decoder $G$ , such that taken together, they learn to represent images with codes from a learned, discrete codebook $\mathcal { Z } = \{ z _ { k } \} _ { k = 1 } ^ { K } \subset \mathbb { R } ^ { n _ { z } }$

precisely, we approximate a given image $x$ by ${ \hat { x } } = G ( z _ { \mathbf { q } } )$ . We obtain $z _ { \mathbf { q } }$ using the encoding $\hat { z } = { E } ( { x } ) \in \mathbb { R } ^ { h \times w \times \bar { n } _ { z } }$ and a subsequent element-wise quantization $\mathbf { q } ( \cdot )$ of each spatial code $\hat { z } _ { i j } \in \mathbb { R } ^ { n _ { z } }$ onto its closest codebook entry $z _ { k }$ :

$$
z _ { \mathbf { q } } = \mathbf { q } ( \boldsymbol { \hat { z } } ) : = \left( \underset { z _ { k } \in \mathcal { Z } } { \arg \operatorname* { m i n } } \lVert \boldsymbol { \hat { z } } _ { i j } - z _ { k } \rVert \right) \in \mathbb { R } ^ { h \times w \times n _ { z } } .
$$

The reconstruction $\hat { x } \approx x$ is then given by

$$
{ \hat { x } } = G ( z _ { \mathbf { q } } ) = G \left( \mathbf { q } ( E ( x ) ) \right) .
$$

Backpropagation through the non-differentiable quantization operation in Eq. (3) is achieved by a straight-through gradient estimator, which simply copies the gradients from the decoder to the encoder [3], such that the model and codebook can be trained end-to-end via the loss function

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { V Q } } ( E , G , \mathcal { Z } ) = \| x - \hat { x } \| ^ { 2 } + \| \mathbf { s g } [ E ( x ) ] - z _ { \mathbf { q } } \| _ { 2 } ^ { 2 } } \\ { + \| \mathbf { s g } [ z _ { \mathbf { q } } ] - E ( x ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

Here, $\mathcal { L } _ { \mathrm { r e c } } = \| x - \hat { x } \| ^ { 2 }$ is a reconstruction loss, $\mathrm { s g } [ \cdot ]$ denotes the stop-gradient operation, and $\lVert \mathbf { s g } [ z _ { \mathbf { q } } ] - E ( x ) \rVert _ { 2 } ^ { 2 }$ is the socalled "commitment loss" [72].

Learning a Perceptually Rich Codebook Using transformers to represent images as a distribution over latent image constituents requires us to push the limits of compression and learn a rich codebook. To do so, we propose $V Q .$ . $G A N _ { ; }$ , a variant of the original VQVAE, and use a discriminator and perceptual loss [40, 30, 39, 17, 47] to keep good perceptual quality at increased compression rate. Note that this is in contrast to previous works which applied pixelbased [71, 61] and transformer-based autoregressive models [8] on top of only a shallow quantization model. More specifically, we replace the $L _ { 2 }$ loss used in [72] for $\mathcal { L } _ { \mathrm { r e c } }$ by a perceptual loss and introduce an adversarial training procedure with a patch-based discriminator $D$ [28] that aims to differentiate between real and reconstructed images:

$$
\mathcal { L } _ { \mathrm { G A N } } ( \{ E , G , \mathcal { Z } \} , D ) = [ \log D ( x ) + \log ( 1 - D ( \hat { x } ) ) ]
$$

The complete objective for finding the optimal compression model $\mathcal { Q } ^ { \ast } = \{ E ^ { \ast } , G ^ { \ast } , \mathcal { Z } ^ { \ast } \}$ then reads

$$
\begin{array} { r l } & { \boldsymbol { \mathcal { Q } } ^ { * } = \underset { E , G , \mathcal { Z } } { \arg \operatorname* { m i n } } \underset { D } { \arg \mathbb { E } } _ { \boldsymbol { x } \sim p ( \boldsymbol { x } ) } \Big [ \mathcal { L } _ { \mathrm { V Q } } ( E , G , \mathcal { Z } ) } \\ & { ~ + \lambda \mathcal { L } _ { \mathrm { G A N } } ( \{ E , G , \mathcal { Z } \} , D ) \Big ] , } \end{array}
$$

where we compute the adaptive weight $\lambda$ according to

$$
\lambda = \frac { \nabla _ { G _ { L } } [ \mathcal { L } _ { \mathrm { r e c } } ] } { \nabla _ { G _ { L } } [ \mathcal { L } _ { \mathrm { G A N } } ] + \delta }
$$

where $\mathcal { L } _ { \mathrm { r e c } }$ is the perceptual reconstruction loss [81], $\nabla _ { G _ { L } } [ \cdot ]$ denotes the gradient of its input w.r.t. the last layer $L$ of the decoder, and $\delta = 1 0 ^ { - 6 }$ is used for numerical stability. To aggregate context from everywhere, we apply a single attention layer on the lowest resolution. This training procedure significantly reduces the sequence length when unrolling the latent code and thereby enables the application of powerful transformer models.

# 3.2. Learning the Composition of Images with Transformers

Latent Transformers With $E$ and $G$ available, we can now represent images in terms of the codebook-indices of their encodings. More precisely, the quantized encoding of an image $x$ is given by $z _ { \mathbf { q } } = \mathbf { q } ( E ( x ) ) \in \mathbb { R } ^ { h \times w \times n _ { z } }$ and is equivalent to a sequence $s \in \{ 0 , . . . , | \mathcal { Z } | - 1 \} ^ { h \times w }$ of indices from the codebook, which is obtained by replacing each code by its index in the codebook $\mathcal { Z }$ :

$$
s _ { i j } = k \mathrm { ~ s u c h ~ t h a t ~ } ( z _ { \mathbf { q } } ) _ { i j } = z _ { k } .
$$

By mapping indices of a sequence $s$ back to their corresponding codebook entries, $z _ { \mathbf { q } } = \left( z _ { s _ { i j } } \right)$ is readily recovered and decoded to an image ${ \hat { x } } = G ( z _ { \mathbf { q } } )$ .

Thus, after choosing some ordering of the indices in $s$ , image-generation can be formulated as autoregressive next-index prediction: Given indices $s _ { < i }$ , the transformer learns to predict the distribution of possible next indices, i.e. $p ( s _ { i } | s _ { < i } )$ to compute the likelihood of the full representation as $\begin{array} { r } { p ( s ) = \prod _ { i } p ( s _ { i } | s _ { < i } ) } \end{array}$ This allows us to directly maximize the log-likelihood of the data representations:

$$
\mathcal { L } _ { \mathrm { T r a n s f o r m e r } } = \mathbb { E } _ { x \sim p ( x ) } \left[ - \log p ( s ) \right] .
$$

Conditioned Synthesis In many image synthesis tasks a user demands control over the generation process by providing additional information from which an example shall be synthesized. This information, which we will call $c$ , could be a single label describing the overall image class or even another image itself. The task is then to learn the likelihood of the sequence given this information $c$ :

$$
p ( s | c ) = \prod _ { i } p ( s _ { i } | s _ { < i } , c ) .
$$

If the conditioning information $c$ has spatial extent, we first learn another $V Q G A N$ to obtain again an index-based representation $r \in \{ 0 , \ldots , | \mathcal { Z } _ { c } | - 1 \} ^ { h _ { c } \times w _ { c } }$ with the newly obtained codebook $\mathcal { Z } _ { c }$ Due to the autoregressive structure of the transformer, we can then simply prepend $r$ to $s$ and restrict the computation of the negative log-likelihood to entries $p ( s _ { i } | s _ { < i } , r )$ . This "decoder-only" strategy has also been successfully used for text-summarization tasks [44].

Generating High-Resolution Images The attention mechanism of the transformer puts limits on the sequence length $h \cdot w$ of its inputs $s$ . While we can adapt the number of downsampling blocks $m$ of our $V Q G A N$ to reduce images of size $H \times W$ to $h = H / 2 ^ { m } \times w = W / 2 ^ { m }$ , we observe degradation of the reconstruction quality beyond a critical value of $m$ , which depends on the considered dataset. To generate images in the megapixel regime, we therefore have to work patch-wise and crop images to restrict the length of $s$ to a maximally feasible size during training. To sample images, we then use the transformer in a sliding-window manner as illustrated in Fig. 3. Our $V Q G A N$ ensures that the available context is still sufficient to faithfully model images, as long as either the statistics of the dataset are approximately spatially invariant or spatial conditioning information is available. In practice, this is not a restrictive requirement, because when it is violated, i.e. unconditional image synthesis on aligned data, we can simply condition on image coordinates, similar to [42].

![](images/3.jpg)  
Figure 3. Sliding attention window.

# 4. Experiments

This section evaluates the ability of our approach to retain the advantages of transformers over their convolutional counterparts (Sec. 4.1) while integrating the effectiveness of convolutional architectures to enable high-resolution image synthesis (Sec. 4.2). Furthermore, in Sec. 4.3, we investigate how codebook quality affects our approach. We close the analysis by providing a quantitative comparison to a wide range of existing approches for generative image synthesis in Sec. 4.4. Based on initial experiments, we usually set $\vert \mathcal { Z } \vert = 1 0 2 4$ and train all subsequent transformer models to predict sequences of length $1 6 \cdot 1 6$ , as this is the maximum feasible length to train a GPT2-medium architecture $3 0 7 { \mathrm { M } }$ parameters) [58] on a GPU with 12GB VRAM. More details on architectures and hyperparameters can be found in the appendix (Tab. 7 and Tab. 8).

# 4.1. Attention Is All You Need in the Latent Space

Transformers show state-of-the-art results on a wide variety of tasks, including autoregressive image modeling. However, evaluations of previous works were limited to transformers working directly on (low-resolution) pixels [55, 12, 26], or to deliberately shallow pixel encodings [8]. This raises the question if our approach retains the advantages of transformers over convolutional approaches.

To answer this question, we use a variety of conditional and unconditional tasks and compare the performance between our transformer-based approach and a convolutional approach. For each task, we train a $V Q G A N$ with $m = 4$ downsampling blocks, and, if needed, another one for the conditioning information, and then train both a transformer and a PixelSNAIL [10] model on the same representations, as the latter has been used in previous state-of-the-art twostage approaches [61]. For a thorough comparison, we vary the model capacities between 85M and 310M parameters and adjust the number of layers in each model to match one another. We observe that PixelSNAIL trains roughly twice as fast as the transformer and thus, for a fair comparison, report the negative log-likelihood both for the same amount of training time ( $P$ -SNAIL time) and for the same amount of training steps ( $P$ -SNAIL steps).

Table 1. Comparing Transformer and PixelSNAIL architectures across different datasets and model sizes. For all settings, transformers outperform the state-of-the-art model from the PixelCNN family, PixelSNAIL in terms of NLL. This holds both when comparing NLL at fixed times (PixelSNAIL trains roughly 2 times faster) and when trained for a fixed number of steps. See Sec. 4.1 for the abbreviations.   

<table><tr><td>Data / # params</td><td>Transformer P-SNAIL steps</td><td>Transformer P-SNAIL time</td><td>PixelSNAIL fixed time</td></tr><tr><td>RIN /85M</td><td>4.78</td><td>4.84</td><td>4.96</td></tr><tr><td>LSUN-CT /310M</td><td>4.63</td><td>4.69</td><td>4.89</td></tr><tr><td>IN / 310M</td><td>4.78</td><td>4.83</td><td>4.96</td></tr><tr><td>D-RIN / 180 M</td><td>4.70</td><td>4.78</td><td>4.88</td></tr><tr><td>S-FLCKR / 310 M</td><td>4.49</td><td>4.57</td><td>4.64</td></tr></table>

Results Tab. 1 reports results for unconditional image modeling on ImageNet (IN) [14], Restricted ImageNet (RIN) [65], consisting of a subset of animal classes from ImageNet, LSUN Churches and Towers (LSUN-CT) [79], and for conditional image modeling of RIN conditioned on depth maps obtained with the approach of [60] (D-RIN) and of landscape images collected from Flickr conditioned on semantic layouts (S-FLCKR) obtained with the approach of [7]. Note that for the semantic layouts, we train the first-stage using a cross-entropy reconstruction loss due to their discrete nature. The results shows that the transformer consistently outperforms PixelSNAIL across all tasks when trained for the same amount of time and the gap increases even further when trained for the same number of steps. These results demonstrate that gains of transformers carry over to our proposed two-stage setting.

# 4.2. A Unified Model for Image Synthesis Tasks

The versatility and generality of the transformer architecture makes it a promising candidate for image synthesis. In the conditional case, additional information $c$ such as class labels or segmentation maps are used and the goal is to learn the distribution of images as described in Eq. (10). Using the same setting as in Sec. 4.1 (i.e. image size $2 5 6 \times 2 5 6$ , latent size $1 6 \times 1 6$ ), we perform various conditional image synthesis experiments:

![](images/4.jpg)  
Figure 4. Transformers within our setting unify a wide range of image synthesis tasks. We show $2 5 6 \times 2 5 6$ synthesis results across different conditioning inputs and datasets, all obtained with the same approach to exploit inductive biases of effective CNN based VQGAN architectures in combination with the expressivity of transformer architectures. Top row: Completions from unconditional training on ImageNet. 2nd row: Depth-to-Image on RIN. 3rd row: Semantically guided synthesis on ADE20K. 4th row: Pose-guided person generation on DeepFashion. Bottom row: Class-conditional samples on RIN.

(i): Semantic image synthesis, where we condition on semantic segmentation masks of ADE20K [83], a webscraped landscapes dataset (S-FLCKR) and COCO-Stuff [6]. Results are depicted in Figure 4, 5 and Fig. 6.

(ii): Structure-to-image, where we use either depth or edge information to synthesize images from both RIN and IN (see Sec. 4.1). The resulting depth-to-image and edge-toimage translations are visualized in Fig. 4 and Fig. 6.

(iii): Pose-guided synthesis: Instead of using the semantically rich information of either segmentation or depth maps, Fig. 4 shows that the same approach as for the previous experiments can be used to build a shape-conditional generative model on the DeepFashion [45] dataset.

(iv): Stochastic superresolution, where low-resolution images serve as the conditioning information and are thereby upsampled. We train our model for an upsampling factor of 8 on ImageNet and show results in Fig. 6.

(v): Class-conditional image synthesis: Here, the conditioning information $c$ is a single index describing the class label of interest. Results for the RIN and IN dataset are demonstrated in Fig. 4 and Fig. 8, respectively.

All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths. In combination, the presented approach can be understood as an efficient, general purpose mechanism for conditional image synthesis. Note that additional results for each experiment can be found in the appendix, Sec. D.

High-Resolution Synthesis The sliding window approach introduced in Sec. 3.2 enables image synthesis beyond a resolution of $2 5 6 \times 2 5 6$ pixels. We evaluate this approach on unconditional image generation on LSUN-CT and FacesHQ (see Sec. 4.3) and conditional synthesis on DRIN, COCO-Stuff and S-FLCKR, where we show results in Fig. 1, 6 and the supplementary (Fig. 29-39). Note that this approach can in principle be used to generate images of arbitrary ratio and size, given that the image statistics of the dataset of interest are approximately spatially invariant or spatial information is available. Impressive results can be achieved by applying this method to image generation from semantic layouts on S-FLCKR, where a strong VQGAN can be learned with $m \ = \ 5$ , so that its codebook together with the conditioning information provides the transformer with enough context for image generation in the megapixel regime.

# 4.3. Building Context-Rich Vocabularies

How important are context-rich vocabularies? To investigate this question, we ran experiments where the transformer architecture is kept fixed while the amount of context encoded into the representation of the first stage is varied through the number of downsampling blocks of our $V Q$ . GAN. We specify the amount of context encoded in terms of reduction factor in the side-length between image inputs and the resulting representations, i.e. a first stage encoding images of size $H \times W$ into discrete codes of size $H / f \times W / f$ is denoted by a factor $f$ For $f = 1$ , we reproduce the approach of [8] and replace our $V Q G A N$ by a $\mathbf { k }$ -means clustering of RGB values with $k = 5 1 2$ .

During training, we always crop images to obtain inputs of size $1 6 \times 1 6$ for the transformer, i.e. when modeling images with a factor $f$ in the first stage, we use crops of size $1 6 f \times 1 6 f$ . To sample from the models, we always apply them in a sliding window manner as described in Sec. 3.

Results Fig. 7 shows results for unconditional synthesis of faces on FacesHQ, the combination of CelebA-HQ [31] and

![](images/5.jpg)  
Figure 5. Samples generated from semantic layouts on S-FLCKR. Sizes from top-to-bottom: $1 2 8 0 \times 8 3 2$ , $1 0 2 4 \times 4 1 6$ and $1 2 8 0 \times$ 240 pixels. Best viewed zoomed in. A larger visualization can be found in the appendix, see Fig 29.

FFHQ [33]. It clearly demonstrates the benefits of powerful VQGANs by increasing the effective receptive field of the transformer. For small receptive fields, or equivalently small $f$ , the model cannot capture coherent structures. For an intermediate value of $f \ = \ 8$ , the overall structure of images can be approximated, but inconsistencies of facial features such as a half-bearded face and of viewpoints in different parts of the image arise. Only our full setting of $f = 1 6$ can synthesize high-fidelity samples. For analogous results in the conditional setting on S-FLCKR, we refer to the appendix (Fig. 13 and Sec. C).

To assess the effectiveness of our approach quantitatively, we compare results between training a transformer directly on pixels, and training it on top of a VQGAN's latent code with $f = 2$ , given a fixed computational budget. Again, we follow [8] and learn a dictionary of 512 RGB values on CIFAR10 to operate directly on pixel space and train the same transformer architecture on top of our VQGAN with a latent code of size $1 6 \times 1 6 = 2 5 6$ We observe improvements of $1 8 . 6 3 \%$ for FIDs and $1 4 . 0 8 \times$ faster sampling of images.

![](images/6.jpg)

Figure 6. Applying the sliding attention window approach (Fig. 3) to various conditional image synthesis tasks. Top: Depth-to-image on RIN, 2nd row: Stochastic superresolution on IN, 3rd and 4th row: Semantic synthesis on S-FLCKR, bottom: Edge-guided synthesis on IN. The resulting images vary between $3 6 8 \times 4 9 6$ and $1 0 2 4 \times 5 7 6$ , hence they are best viewed zoomed in.   

<table><tr><td>Dataset</td><td>ours</td><td>SPADE [53]</td><td>Pix2PixHD (+aug) [75]</td><td>CRN [9]</td></tr><tr><td>COCO-Stuff</td><td>22.4</td><td>22.6/23.9(*)</td><td>111.5 (54.2)</td><td>70.4</td></tr><tr><td>ADE20K</td><td>35.5</td><td>33.9/35.7(*)</td><td>81.8 (41.5)</td><td>73.3</td></tr></table>

Table 2. FID score comparison for semantic image synthesis $( 2 5 6 \times 2 5 6$ pixels). (\*): Recalculated with our evaluation protocol based on [50] on the validation splits of each dataset.

# 4.4. Benchmarking Image Synthesis Results

In this section we investigate how our approach quantitatively compares to existing models for generative image synthesis. In particular, we assess the performance of our model in terms of FID and compare to a variety of established models (GANs, VAEs, Flows, AR, Hybrid). The results on semantic synthesis are shown in Tab. 2, where we compare to [53, 75, 35, 9], and the results on unconditional face synthesis are shown in Tab. 3. While some task-specialized GAN models report better FID scores, our approach provides a unified model that works well across a wide range of tasks while retaining the ability to encode and reconstruct images. It thereby bridges the gap between purely adversarial and likelihood-based approaches.

![](images/7.jpg)  
io Q-FlAHHQ $\left| s \right| = 1 6 { \cdot } 1 6 =$ $t = 1 . 0$ and top $k$ sampling with $k = 1 0 0$ .Last row reports the speedup over the f1 baseline which operates directly on pixels and takes 7258 seconds to produce a sample on a NVIDIA GeForce GTX Titan X.

<table><tr><td colspan="2">CelebA-HQ 256 × 256</td><td colspan="2">FFHQ 256 × 256</td></tr><tr><td>Method</td><td>FID ↓</td><td>Method</td><td>FID ↓</td></tr><tr><td>GLOW [37]</td><td>69.0</td><td>VDVAE (t = 0.7) [11]</td><td>38.8</td></tr><tr><td>NVAE [69]</td><td>40.3</td><td>VDVAE ( = 1.0)</td><td>33.5</td></tr><tr><td>PIONEER (B.) [23]</td><td>39.2 (25.3)</td><td>VDVAE (t = 0.8)</td><td>29.8</td></tr><tr><td>NCPVAE [1]</td><td>24.8</td><td>VDVAE (t = 0.9)</td><td>28.5</td></tr><tr><td>VAEBM [77]</td><td>20.4</td><td>VQGAN+P.SNAIL</td><td>21.9</td></tr><tr><td>Style ALAE [56]</td><td>19.2</td><td>BigGAN</td><td>12.4</td></tr><tr><td>DC-VAE [54]</td><td>15.8</td><td>ours (k=300)</td><td>9.6</td></tr><tr><td>ours (k=400)</td><td>10.2</td><td>U-Net GAN (+aug) [66]</td><td>10.9 (7.6)</td></tr><tr><td>PGGAN [31]</td><td>8.0</td><td>StyleGAN2 (+aug) [34]</td><td>3.8 (3.6)</td></tr></table>

Table 3. FID score comparison for face image synthesis. CelebAHQ results reproduced from [1, 54, 77, 24], FFHQ from [66, 32].

Autoregressive models are typically sampled with a decoding strategy [27] such as beam-search, top- $\mathbf { \nabla } \cdot \mathbf { k }$ or nucleus sampling. For most of our results, including those in Tab. 2, we use top- $\mathbf { \nabla } \cdot \mathbf { k }$ sampling with $k = 1 0 0$ unless stated otherwise. For the results on face synthesis in Tab. 3, we computed scores for $k \in \{ 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 , 5 0 0 \}$ and report the best results, obtained with $k = 4 0 0$ for CelebA-HQ and $k = 3 0 0$ for FFHQ. Fig. 10 in the supplementary shows FID and Inception scores as a function of $k$ .

Class-Conditional Synthesis on ImageNet To address a direct comparison with the previous state-of-the-art for autoregressive modeling of class-conditional image synthesis on ImageNet, VQVAE-2 [61], we train a class-conditional ImageNet transformer on $2 5 6 \times 2 5 6$ images, using a VQ$G A N$ with $\dim { \mathcal { Z } } \ = \ 1 6 3 8 4$ and $f \ = \ 1 6$ , and additionally compare to BigGAN [4], IDDPM [49], DCTransformer [48] and ADM [15] in Tab. 4. Note that our model uses $\simeq 1 0 \times$ less parameters than VQVAE-2, which has an estimated parameter count of 13.5B (estimate based on [67]).

Samples of this model for different ImageNet classes are shown in Fig. 8. We observe that the adversarial training of the corresponding VQGAN enables sampling of highquality images with realistic textures, of comparable or higher quality than existing approaches such as BigGAN and VQVAE-2, see also Fig. 14-17 in the supplementary.

<table><tr><td>Model</td><td>acceptance rate</td><td>FID</td><td>IS</td></tr><tr><td>mixed k, p = 1.0</td><td>1.0</td><td>17.04</td><td>70.6 ± 1.8</td></tr><tr><td>k = 973, p = 1.0</td><td>1.0</td><td>29.20</td><td>47.3 ± 1.3</td></tr><tr><td>k = 250, p = 1.0</td><td>1.0</td><td>15.98</td><td>78.6 ± 1.1</td></tr><tr><td>k = 973, p = 0.88</td><td>1.0</td><td>15.78</td><td>74.3 ± 1.8</td></tr><tr><td>k = 600, p = 1.0</td><td>0.05</td><td>5.20</td><td>280.3 ± 5.5</td></tr><tr><td>mixed k, p = 1.0</td><td>0.5</td><td>10.26</td><td>125.5 ± 2.4</td></tr><tr><td>mixed k, p = 1.0</td><td>0.25</td><td>7.35</td><td>188.6 ± 3.3</td></tr><tr><td>mixed k, p = 1.0</td><td>0.05</td><td>5.88</td><td>304.8 ± 3.6</td></tr><tr><td>mixed k, p = 1.0</td><td>0.005</td><td>6.59</td><td>402.7 ± 2.9</td></tr><tr><td>DCTransformer [48]</td><td>1.0</td><td>36.5</td><td>n/a</td></tr><tr><td>VQVAE-2 [61]</td><td>1.0</td><td>∼31</td><td>~45</td></tr><tr><td>VQVAE-2</td><td>n/a</td><td>∼10</td><td>∼330</td></tr><tr><td>BigGAN [4]</td><td>1.0</td><td>7.53</td><td>168.6 ± 2.5</td></tr><tr><td>BigGAN-deep</td><td>1.0</td><td>6.84</td><td>203.6 ± 2.6</td></tr><tr><td>IDDPM [49]</td><td>1.0</td><td>12.3</td><td>n/a</td></tr><tr><td>ADM-G, no guid. [15]</td><td>1.0</td><td>10.94</td><td>100.98</td></tr><tr><td></td><td>1.0</td><td>4.59</td><td>186.7</td></tr><tr><td>ADM-G, 1.0 guid. ADM-G, 10.0 guid.</td><td>1.0</td><td>9.11</td><td>283.92</td></tr><tr><td>val. data</td><td>1.0</td><td>1.62</td><td>234.0 ± 3.9</td></tr></table>

Table 4. FID score comparison for class-conditional synthesis on $2 5 6 \times 2 5 6$ ImageNet, evaluated between 50k samples and the training split. Classifier-based rejection sampling as in VQVAE-2 uses a ResNet-101 [22] classifier. BigGAN(-deep) evaluated via https: / /t fhub.dev/deepmind truncated at 1.0. "Mixed" $k$ refers to samples generated with different top- $\mathbf { \nabla } \cdot \mathbf { k }$ values, here $k \in$ $\{ 1 0 0 , 2 0 0 , 2 5 0 , 3 0 0 , 3 5 0 , 4 0 0 , 5 0 0 , 6 0 0 , 8 0 0 , 9 7 3 \} .$ .

Quantitative results are summarized in Tab. 4. We report FID and Inception Scores for the best $k / p$ in top-k/top-p sampling. Following [61], we can further increase quality via classifier-rejection, which keeps only the best $m$ -outof $^ n$ samples in terms of the classifier's score, i.e. with an acceptance rate of $m / n$ . We use a ResNet-101 classifier [22].

We observe that our model outperforms other autoregressive approaches (VQVAE-2, DCTransformer) in terms of FID and IS, surpasses BigGAN and IDDPM even for low rejection rates and yields scores close to the state of the art for higher rejection rates, see also Fig. 9.

How good is the VQGAN? Reconstruction FIDs obtained via the codebook provide an estimate on the achievable FID of the generative model trained on it. To quantify the performance gains of our VQGAN over discrete VAEs trained without perceptual and adversarial losses (e.g. VQVAE-2, DALL-E [59]), we evaluate this metric on ImageNet and report results in Tab. 5. Our VQGAN outperforms nonadversarial models while providing significantly more compression (seq. length of 256 vs. $5 1 2 0 = 3 2 ^ { 2 } + 6 4 ^ { 2 }$ for VQVAE-2, 256 vs 1024 for DALL-E). As expected, larger versions of VQGAN (either in terms of larger codebook sizes or increased code lengths) further improve performance. Using the same hierarchical codebook setting as in VQVAE-2 with our model provides the best reconstruction FID, albeit at the cost of a very long and thus impractical sequence. The qualitative comparison corresponding to the results in Tab. 5 can be found in Fig. 12.

![](images/8.jpg)  
Figure 8. Samples from our class-conditional ImageNet model trained on $2 5 6 \times 2 5 6$ images

![](images/9.jpg)  
Figure 9. FID and Inception Score as a function of top-k, nucleus and rejection filtering.

Table 5. FID on ImageNet between reconstructed validation split and original validation (FID/val) and training (FID/train) splits. \*trained with Gumbel-Softmax reparameterization as in [59, 29].   

<table><tr><td>Model</td><td>Codebook Size</td><td>dim Z</td><td>FID/val</td><td>FID/train</td></tr><tr><td>VQVAE-2</td><td>64 × 64 &amp; 32 × 32</td><td>512</td><td>n/a</td><td>∼ 10</td></tr><tr><td>DALL-E [59]</td><td>32 × 32</td><td>8192</td><td>32.01</td><td>33.88</td></tr><tr><td>VQGAN</td><td>16 × 16</td><td>1024</td><td>7.94</td><td>10.54</td></tr><tr><td>VQGAN</td><td>16 × 16</td><td>16384</td><td>4.98</td><td>7.41</td></tr><tr><td>VQGAN*</td><td>32 × 32</td><td>8192</td><td>1.49</td><td>3.24</td></tr><tr><td>VQGAN</td><td>64 × 64 &amp; 32 × 32</td><td>512</td><td>1.45</td><td>2.78</td></tr></table>

# 5. Conclusion

This paper adressed the fundamental challenges that previously confined transformers to low-resolution images. We proposed an approach which represents images as a composition of perceptually rich image constituents and thereby overcomes the infeasible quadratic complexity when modeling images directly in pixel space. Modeling constituents with a CNN architecture and their compositions with a transformer architecture taps into the full potential of their complementary strengths and thereby allowed us to represent the first results on high-resolution image synthesis with a transformer-based architecture. In experiments, our approach demonstrates the efficiency of convolutional inductive biases and the expressivity of transformers by synthesizing images in the megapixel range and outperforming state-of-the-art convolutional approaches. Equipped with a general mechanism for conditional synthesis, it offers many opportunities for novel neural rendering approaches.

# Taming Transformers for High-Resolution Image Synthesis

Supplementary Material

T F x u S.  .

# A. Changelog

We summarize changes between this version 1 of the paper and its previous version 2.

In the previous version, Eq. (4) had a weighting term $\beta$ on the commitment loss, and Tab. 8 reported a value of $\beta = 0 . 2 5$ for all models. However, due to a bug in the implementation, $\beta$ was never used and all models have been trained with $\beta = 1 . 0$ . Thus, we removed $\beta$ in Eq. (4).

Wudateclass-condital nthe resultn ImageNe S.The revius results, icluhe a. fos sptaheo ec e e e e 16 aulatdovr 8 batches which tok 45.8 days n  sing A100 GPU.The previus model had beentrai for 1.0 million steps. Furthermore, the FID values were based on $5 0 \mathrm { k }$ (18k) samples against 50k (18k) training examples (to al traini example ImageNet usiy [50. Weudate al qualitative gure howig amp this model and added visualizations of the effect of tuning top- $k / p$ or rejection rate in Fig. 14-26.

T veu wokhl T t r al eul  - e [ ] Ta th wie et BGA BGAN-ee [ e  psul vaablee same codeWe ollow he commo evaluation protocol orclass-conditonal ImageNet syntheis from [4] and ealuate $5 0 \mathrm { k }$ $2 5 6 \times 2 5 6$ using Pillow [73]. FID and Inception Scores are then computed with torch-fidelity [50].

W u also added Fig. 10, which visualizes the effect of tuning $k$ in top- $\mathbf { \nabla } \cdot \mathbf { k }$ sampling on FID and IS.

# B. Implementation Details

Theype   h  pa pe l . Except for the $c$ IN (big), COCO-Stuff and ADE20K models, these hyperparameters are set such that each transformer model can be rai wita batc-sizat least   GPU wi12GB VRAM, but ee trainn-4 GUs w accumulated VRAM of 48 GB. If hardware permits, 16-bit precision training is enabled.

<table><tr><td>Dataset</td><td>ours-previous (+R)</td><td>BigGAN (-deep)</td><td>MSP</td></tr><tr><td>IN 256, 50K</td><td>19.8 (11.2)</td><td>7.1 (7.3)</td><td>n.a.</td></tr><tr><td>IN 256, 18K</td><td>23.5</td><td>9.6 (9.7)</td><td>50.4</td></tr></table>

<table><tr><td>Dataset</td><td>ours-previous</td><td>ours-new</td></tr><tr><td>CelebA-HQ 256</td><td>10.7</td><td>10.2</td></tr><tr><td>FFHQ 256</td><td>11.4</td><td>9.6</td></tr></table>

implementation compared to the new implementation. See also Tab. 3 for comparison with other methods.

![](images/10.jpg)  
Figure 10.FID and Inception Score as a function of to $\mathbf { \nabla } \cdot \mathbf { k }$ for CelebA-HQ (left) and FFHQ (right).

<table><tr><td>Encoder</td><td>Decoder</td></tr><tr><td>x  R×W×C</td><td>zq  Rh×w×nz</td></tr><tr><td>Conv2D → RH×W ×C′</td><td>Conv2D → Rh×w×C&quot;</td></tr><tr><td>m× { Residual Block, Downsample Block} → Rh×w×C&quot;′</td><td>Residual Block → Rh×w×C′</td></tr><tr><td>Residual Block → Rh ×w×C&quot;</td><td>Non-Local Block → Rh× w×C&quot;</td></tr><tr><td>Non-Local Block → Rh×w×C′</td><td>Residual Block → Rh× w×C&quot;</td></tr><tr><td>Residual Block → Rh×w×C&quot;</td><td>m× { Residual Block, Upsample Block} → RH×W ×C′</td></tr><tr><td>GroupNorm, Swish, Conv2D → Rh× w×nz</td><td>GroupNorm, Swish, Conv2D → RH ×W ×C</td></tr></table>

Table 7. High-level architecture of the encoder and decoder of our $\overline { { V Q G A N } }$ The design of the networks follows the architecture presented $\begin{array} { r } { h = \frac { H } { 2 ^ { m } } } \end{array}$ , $\begin{array} { r } { w = \frac { W } { 2 ^ { m } } } \end{array}$ and $f = 2 ^ { m }$

Veu ol  ne is described in Tab. 7. Note that we adopt the compression rate by tuning the number of downsampling steps $m$ .Further note that $\lambda$ reconstructions. As a rule of thumb, we recommend setting $\lambda = 0$ for at least one epoch.

Tuel  u    y $t = 1 . 0$ and a top $k$ cutoff at $k = 1 0 0$ (with higher top- $k$ values for larger codebooks).

# C. On Context-Rich Vocabularies

Sec. 4.3 investigated the effect of the downsampling factor $f$ used for encoding images. As demonstrated in Fig. 7, large since larger $f$ correspond to larger compression rates, the reconstruction quality of the $V Q G A N$ starts to decrease after a negative log-likelihood obtained by the transformer for values of $f$ ranging from 1 to 64. The latter provides a measure of the ability to model the distribution of the image representation, which increases with $f$ .The reconstruction error on the other hand decreases with $f$ and the qualitative results on the right part show that beyond a critical value of $f$ , in this case $f = 1 6$ , bound on the quality that can be achieved.

Hence, Fig. 11 shows that we must learn perceptually rich encodings, i.e. encodings with a large $f$ and perceptually faithful [ used in DALL-E [59]. We observe that for $f = 8$ and 8192 codebook entries, both the VQVAE and VQGAN capture the $V Q G A N$ further to $f = 1 6$ , we see that some reconstructed parts are not perfectly aligned with the input anymore (e.g. the how the $V Q G A N$ provides high-fidelity reconstructions at large factors, and thereby enables efficient high-resolution image synthesis with transformers.

Table 8. Hyperparameters. For every experiment, we set the number of attention heads in the transformer to $\overline { { n _ { h } = 1 6 } }$ . $n _ { l a y e r }$ denotes the number of transformer blocks, # params the number of transformer parameters, $n _ { z }$ the dimensionality of codebook entries, $| \mathcal { Z } |$ the number $n _ { e }$ the embedding dimensionality and $m$ the number of downsampling steps in the VQGAN. D-RINv1 is the experiment which compares to Pixel-SNAIL in Sec. 4.1. Note that the experiment (FacesHQ, $f = 1 ) ^ { * }$ does not use a learned $V Q G A N$ but a fixed $\mathbf { k }$ means clustering algorithm as in [8] with $K = 5 1 2$ centroids. A prefix "c" refers to a class-conditional model. The models marked with a $^ \bullet \ast ^ { \bullet }$ are trained on the same VQGAN.   

<table><tr><td>Experiment</td><td>nlayer</td><td># params</td><td>[M] nz</td><td>|Z|</td><td>dropout</td><td>length(s)</td><td>ne</td><td>m</td></tr><tr><td>RIN</td><td>12</td><td>85</td><td>64</td><td>768</td><td>0.0</td><td>512</td><td>1024</td><td>4</td></tr><tr><td>c-RIN</td><td>18</td><td>128</td><td>64</td><td>768</td><td>0.0</td><td>257</td><td>768</td><td>4</td></tr><tr><td>D-RINv1</td><td>14</td><td>180</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>768</td><td>4</td></tr><tr><td>D-RINv2</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>4</td></tr><tr><td>IN</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>256</td><td>1024</td><td>4</td></tr><tr><td>c-IN</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>257</td><td>1024</td><td>4</td></tr><tr><td>c-IN (big)</td><td>48</td><td>1400</td><td>256</td><td>16384</td><td>0.0</td><td>257</td><td>1536</td><td>4</td></tr><tr><td>IN-Edges</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>3</td></tr><tr><td>IN-SR</td><td>12</td><td>153</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>3</td></tr><tr><td>S-FLCKR, f = 4</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>2</td></tr><tr><td>S-FLCKR, f = 16</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>4</td></tr><tr><td>S-FLCKR, f = 32</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>5</td></tr><tr><td>(FacesHQ, f = 1)*</td><td>24</td><td>307</td><td></td><td>512</td><td>0.0</td><td>512</td><td>1024</td><td></td></tr><tr><td>FacesHQ, f = 2</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>1</td></tr><tr><td>FacesHQ, f = 4</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>2</td></tr><tr><td>FacesHQ, f = 8</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>3</td></tr><tr><td>FacesHQ**, f = 16</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>512</td><td>1024</td><td>4</td></tr><tr><td>FFHQ**, f = 16</td><td>28</td><td>355</td><td>256</td><td>1024</td><td>0.0</td><td>256</td><td>1024</td><td>4</td></tr><tr><td>CelebA-HQ**, f = 16</td><td>28</td><td>355</td><td>256</td><td>1024</td><td>0.0</td><td>256</td><td>1024</td><td>4</td></tr><tr><td>FFHQ (big)</td><td>24</td><td>801</td><td>256</td><td>1024</td><td>0.0</td><td>256</td><td>1664</td><td>4</td></tr><tr><td>CelebA-HQ (big)</td><td>24</td><td>801</td><td>256</td><td>1024</td><td>0.0</td><td>256</td><td>1664</td><td>4</td></tr><tr><td>COCO-Stuff</td><td>32</td><td>651</td><td>256</td><td>8192</td><td>0.0</td><td>512</td><td>1280</td><td>4</td></tr><tr><td>ADE20K</td><td>28</td><td>405</td><td>256</td><td>4096</td><td>0.1</td><td>512</td><td>1024</td><td>4</td></tr><tr><td>DeepFashion</td><td>18</td><td>129</td><td>256</td><td>1024</td><td>0.0</td><td>340</td><td>768</td><td>4</td></tr><tr><td>LSUN-CT</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>256</td><td>1024</td><td>4</td></tr><tr><td>CIFAR-10</td><td>24</td><td>307</td><td>256</td><td>1024</td><td>0.0</td><td>256</td><td>1024</td><td>1</td></tr></table>

To illustrate how the choice of $f$ depends on the dataset, Fig. 13 presents results on S-FLCKR. In the left part, it shows, analogous to Fig. 7, how the quality of samples increases with increasing $f$ . However, in the right part, it shows that reconstructions remain faithful perceptually faithful even for $f 3 2$ , which is in contrast to the corresponding results on faces allow us to generate high-resolution landscapes even more efficiently with $f = 3 2$ .

# D. Additional Results

iv e prome repeivelyorBigA we prouhle he proimodeSmil li can be found in Fig. 40 (ADE20K) and Fig. 41 (COCO-Stuff)6.

Coarn I Trevala eiveupc, pa ravo o, I [B tu t a resolution of $1 9 2 \times 1 9 2$ .As our approach leverages a strong compression method to obtain context-rich representations oage the lear ansorme mode e n nthez magemuchhe eoluWecpa aeheico plo ho a tnthezconsstent copleons aaicalyc fidelyThesu 8 baiom openai.com/blog/image-gpt/.

Aial High-ResolutionResul Fi.  30 1 andFi contaiial HR resul n he-FLCKRa for both $f = 1 6$ $m = 4 ,$ and $f = 3 2$ $( m = 5$ (semantically guided). In particular, we provide an enlarged version of Fig. 5 haseolui  ep  nce on S-FLCKR in Fig. 37 with $f = 1 6$ and in Fig. 38 with $f = 3 2$ , and unconditional image generation on LSUN-CT (see Sec. 4.1) in Fig. 39. Moreover, for images of size $2 5 6 \times 2 5 6$ , we provide results for generation from semantic layout on (Eind CCStufF1eth-o-  i puiere Fig. 43 and class-conditional synthesis on RIN in Fig. 44.

# E. Nearest Neighbors of Samples

O a twcoit   tas hebe aatn  t nd ec CelAQ, respeively an anohe or he best taig NLL (at epo 100.We hen prod sams ro h as hha traieampes wher mp omeckpoint wih bestaliatin LL e valLLdeicc whi are not found in the training data.

Baeua FID scors o 3.86 on CelebA-Q and 2.68on FFHQ, compare to 10.2 and 9.6 or the best val. NLL checpointsWhil the training data.

Oucass-odal maNee ot ispayverilLL, nhe shown in Fig. 46 also provide evidence that the model produces new, high-quality samples.

# F. On the Ordering of Image Representations

F  a  k h ano e centered object. (iii) $\mathbf { z }$ -curve, also known as $z$ -order or morton curve, which introduces the prior of preserved locality when mpp ereta ntessl h recoons for each permutation in a controlled setting, i.e. we fix initialization and computational budget.

Ru l t O ordering [71, 8] outperforms other orderings.

![](images/11.jpg)  
factors $f$ suffer after a critical value (here, $f = 1 6$ ). For more details, see Sec. C.

![](images/12.jpg)  
F

![](images/13.jpg)  
Figure 13. Samples on landscape dataset (left) obtained with different factors $\overline { { f } }$ , analogous to Fig. 7. In contrast to faces, a factor of $\overline { { f = 3 2 } }$ still allows for faithful reconstructions (right). See also Sec. C.

![](images/14.jpg)  
salamandr (top) and 97: drake (bottom). We report class labels as in VQVAE-2 [61].

![](images/15.jpg)  
anemone (top) and 141: redshank (bottom). We report class labels as in VQVAE-2 [61].

![](images/16.jpg)  
(top) and 22: bald eagle (bottom).

![](images/17.jpg)  
and 9: ostrich (bottom).

![](images/18.jpg)  
acc. rate 1.0

![](images/19.jpg)  
933: cheeseburger acc. rate 0.5

![](images/20.jpg)  
acc. rate 0.1

992: agaric acc. rate 0.5 acc. rate 0.1 acc. rate 1.0 acc. rate 0.1

![](images/21.jpg)  
acc. rate 1.0

![](images/22.jpg)

![](images/23.jpg)

![](images/24.jpg)  
200: tibetian terrier acc. rate 0.5   
recognizable objects compared to the unguided samples. Here, $k = 9 7 3$ , $p = 1 . 0$ are fixed for all samples. Note that $k = 9 7 3$ is the effective size of the VQGAN's codebook, i.e. it describes how many entries of the codebook with $\operatorname* { l i m } \mathcal { Z } = 1 6 3 8 4$ are actually used.

![](images/25.jpg)  
Figure 19. Visualizing the effect of varying $\overline { { k } }$ t- p . t e paby tun a tkn y aResNet-101 classifer trained on ImageNet and samples fromour class-conditional ImageNet model. Lower values $k$ produce more uniform, low-entropic images compared to samples obtained with full $k$ Here, an acceptance rate of 1.0 and $p = 1 . 0$ are fixed for all samples. Note that $k = 9 7 3$ is the effective size of the VQGAN's codebook, i.e. it describes how many entries of the codebook with din $1 \mathcal { Z } = 1 6 3 8 4$ are actually used.

![](images/26.jpg)  
Figure 20. Visualizing the effect of varying $p$ in top-p sampling (or nucleus sampling [27]) by using a ResNet-101 classifier trained on ImageNet and samples from our class-conditional ImageNet model. Lowering $p$ has similar effects as decreasing $k$ , see Fig. 19. Here, an acceptance rate of 1.0 and $k = 9 7 3$ are fixed for all samples.

![](images/27.jpg)  
Figure 21. Random samples on $2 5 6 \times 2 5 6$ class-conditional ImageNet with $k \in$ [100, 200, 250, 300, 350, 400, 500, 600, 800, 973] , $p = 1 . 0$ , acceptance rate 1.0. FID: 17.04, IS: $7 0 . 6 \pm 1 . 8$ .Please see https: / /git . io/JLlvY for an uncompressed version. 23

![](images/28.jpg)  
Figure 22. Random samples on $2 5 6 \times 2 5 6$ class-conditional ImageNet with $k = 6 0 0$ $p = 1 . 0$ , acceptance rate 0.05. FID: 5.20, IS: $2 8 0 . 3 \pm 5 . 5$ Please see https: / /git . io/JLlvY for an uncompressed version. 24

![](images/29.jpg)  
Figure 23. Random samples on $2 5 6 \times 2 5 6$ class-conditional ImageNet with $k = 2 5 0$ $p = 1 . 0$ , acceptance rate 1.0. FID: 15.98, IS: $7 8 . 6 \pm 1 . 1$ .Please see https: / /git. io/JLlvY for an uncompressed version. 25

![](images/30.jpg)  
Figure 24. Random samples on $2 5 6 \times 2 5 6$ class-conditional ImageNet with $k = 9 7 3$ , $p = 0 . 8 8$ , acceptance rate 1.0. FID: 15.78, IS: $7 4 . 3 \pm 1 . 8$ Please see https: / /git.io/JLlvY for an uncompressed version. 26

![](images/31.jpg)  
Figure 25. Random samples on $2 5 6 \times 2 5 6$ class-conditional ImageNet with $k \in$ [100, 200, 250, 300, 350, 400, 500, 600, 800, 973] , $p = 1 . 0$ , acceptance rate 0.005. FID: 6.59, IS: $4 0 2 . 7 \pm 2 . 9$ Please see https: / /git.io/JLlvY for an uncompressed version. 27

![](images/32.jpg)  
Figure 26. Random samples on $2 5 6 \times 2 5 6$ class-conditional ImageNet with $k \in$ [100, 200, 250, 300, 350, 400, 500, 600, 800, 973] , $p = 1 . 0$ , acceptance rate 0.05. FID: 5.88, IS: $3 0 4 . 8 \pm 3 . 6$ Please see https: / /git.io/JLlvY for an uncompressed version. 28

![](images/33.jpg)  
Figure 27. Comparing our approach with the pixel-based approach of [8]. Here, we use our $\overline { { f = 1 6 } }$ S-FLCKR model to obtain high-fidelity three of [8] (bottom).

![](images/34.jpg)  
Figure 28. Comparing our approach with the pixel-based approach of [8]. Here, we use our $\overline { { f = 1 6 } }$ S-FLCKR model to obtain high-fidelity three of [8] (bottom).

![](images/35.jpg)  
Figure 29. Samples generated from semantic layouts on S-FLCKR. Sizes from top-to-bottom: $1 2 8 0 \times 8 3 2$ , $1 0 2 4 \times 4 1 6$ and $1 2 8 0 \times 2 4 0$ pixels.

![](images/36.jpg)  
Figure 30. Samples generated from semantic layouts on S-FLCKR. Sizes from top-to-bottom: $1 5 3 6 \times 5 1 2$ , $1 8 4 0 \times 1 0 2 4$ , and $1 5 3 6 \times 6 2 0$ oixels.

![](images/37.jpg)  
Figure 31. Samples generated from semantic layouts on S-FLCKR. Sizes from top-to-bottom: $2 0 4 8 \times 5 1 2$ ,1460 × 440, $2 0 3 2 \times 4 4 8$ and $2 0 1 6 \times 6 7 2$ pixels.

![](images/38.jpg)  
Figure 32. Samples generated from semantic layouts on S-FLCKR. Sizes from top-to-bottom: $1 2 8 0 \times 8 3 2$ , $1 0 2 4 \times 4 1 6$ and $1 2 8 0 \times 2 4 0$ ixels.

![](images/39.jpg)  
Figure 33. Depth-guided neural rendering on RIN with $f = 1 6$ using the sliding attention window.

![](images/40.jpg)  
Figure 34. Depth-guided neural rendering on RIN with $\overline { { f = 1 6 } }$ using the sliding attention window

![](images/41.jpg)  
IN with $f = 8$ , using the sliding attention window.

![](images/42.jpg)  
Figure 36. Additional results for stochastic superresolution with an $\overline { { f = 1 6 } }$ model on IN, using the sliding attention window.

![](images/43.jpg)  
Figure 37. Samples generated from semantic layouts on S-FLCKR with $\overline { { f = 1 6 } }$ , using the sliding attention window.

![](images/44.jpg)  
Figure 38. Samples generated from semantic layouts on S-FLCKR with $\overline { { f = 3 2 } }$ , using the sliding attention window.

![](images/45.jpg)  
F .Uncdial smple fommodel traie  LUN Churce& Towers usighe idittew

![](images/46.jpg)  
Figure 40. Qualitative comparison to [53] on $2 5 6 \times 2 5 6$ images from the ADE20K dataset.

![](images/47.jpg)  
Figure 41. Qualitative comparison to [53] on $2 5 6 \times 2 5 6$ images from the COCO-Stuff dataset.

![](images/48.jpg)  
Figure 42. Conditional samples for the depth-to-image model on IN.

![](images/49.jpg)  
Figure 43. Conditional samples for the pose-guided synthesis model via keypoints on DeepFashion.

![](images/50.jpg)  
Figure 44. Samples produced by the class-conditional model trained on RIN.

![](images/51.jpg)  
Figure 45. Nearest neighbors for our face-models trained on FFHQ and CelebA-HQ $\overline { { \mathrm { 2 5 6 \times 2 5 6 ~ p i x } } }$ , based on the LPIPS [82] distance. ca . See also Sec. E.

![](images/52.jpg)  
Figure 46. Nearest neighbors for our class-conditional ImageNet model $\overline { { ( 2 5 6 \times 2 5 6 ~ \mathrm { p i x } ) } }$ , based on the LPIPS [82] distance. The left c

![](images/53.jpg)  
Figure 47. Top: All sequence permutations we investigate, illustrated on a $4 \times 4$ grid. Bottom: The transformer architecture is permutation di

![](images/54.jpg)  
.

# References

CoRR, abs/2010.02917, 2020. 8   
2   
conditional computation. CoRR, abs/1308.3432, 2013. 4 International Conference on Learning Representations, ICLR, 2019. 8, 10, 16, 17, 18, 19   
[ Bn Banic M, p,  e ShyaGri SasydskendigaralrielHerbertossGeterue Tm ean Rw a B Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165, 2020. 1 recognition (CVPR), 2018 IEEE conference on. IEEE, 2018. 6   
-h ur De ale on nFul  tal Intelligence, 2018. 5 iH pretraining from pixels. 2020. 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 29, 30   
o   - cerVisn, ICC01Veice, Ial October,01 pa 515ur Socie 7   
[0 X i I ICML, volume 80 of Proceedings of Machine Learning Research, pages 863871. PMLR, 2018. 2, 5   
[  /0. 8   
i   
, ICLR, 2019. 2   
IEEE Computer Society Conference on Computer Vision and Pattern Recognition CVPR, 2009. 5   
[15] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis, 2021. 8, 10   
o, MaH scale. 2020. 1   
inNeural Ioain rocessiyste al oerecnNeural Ioainrocessiystes eur sentations. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, 2020. 2   
, abs/1903.04933, 2019. 16, 17, 18, 19   
0u eaivveraleva eural Ioai rocssteal onec Information Processing Systems, NeurIPS, 2014. 2   
y images on a small compute budget. CoRR, abs/2009.04433, 2020. 2   
[ e X Zd, 2015.8 HoneoiiVis Perth, Australia, December 2-6, 2018, Revised Selected Papers, Part I, 2018. 8   
[e Win eplis  VOS 1-5, 2020, pages 31093118. IEEE, 2020. 8   
[\~ xjuy   
[6JoaHo a   sual, abs/1912.12180, 2019. 2, 5   
o view.net, 2020. 8, 22   
. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2017. 4, 11   
, 2016.9 volume 9906 of Lecture Notes in Computer Science, pages 694711. Springer, 2016. 4   
CoRR, abs/1710.10196, 2017. 6, 8   
Bieual ao ul ei 2020, December 6-12, 2020, virtual, 2020. 8   
T Vi RiVPR019 LoBac AUSA J-01, Computer Vision Foundation / IEEE, 2019. 7   
ly tt0Sa June 13-19, 2020, pages 81078116. IEEE, 2020. 8   
[ah   
[36] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks, 2017. 2   
ivvo Inormation Processing Systems 31:Annual ConferenceonNeural Information Processin Systems 2018, NeurIPS, 2018.8   
tions, ICLR, 2014. 2   
, 2016. 4   
[0   Sø  øy Hug    i learned similarity metric, 2015. 4   
67066713. AAAI Press, 2019. 2   
[Hnehn a-Cua  dzoe by parts via conditional coordinating. In ICCV, pages 45114520. IEEE, 2019. 5   
o , 2019. 2   
by summarizing long sequences. In ICLR (Poster). OpenReview.net, 2018. 4   
is. I roi  oecn r  ateci PR), u1   
. InIntial one LearRereis, CLR019, ew rleans, L,USA May , 1OpeRevi, 2019. 14   
  
[49] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models, 2021. 8, 10   
0 metrics for generative models in pytorch, 2020. Version: 0.3.0, D0I: 10.5281/zenodo.4957738. 7, 10 and Pattern Recognition, pages 18, 2007. 2   
inference, 2016. 2   
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2019. 2, 7, 42, 43   
[4 Gurarmar Dache Li KwonjonLe, ndZuwe Tu.Dualritnciv eraivuter 02   
In ICML, volume 80 of Proceedings of Machine Learning Research, pages 40524061. PMLR, 2018. 2, 3, 5   
s o C atR VPR2020SatWAUSA Jun-,020 pa092101EEE   
[57] A. Radford. Improving language understanding by generative pre-training. 2018. 1   
learners. 2019. 1, 5, 11   
e text-to-image generation, 2021. 9, 10, 12   
[0   : 5   
18, 19   
[ I 2014. 2   
V ooi en oereceGlasUKAugus - 2 on, ar VIolum36 LeceNotes Science, pages 647664. Springer, 2020. 2   
[ . volume 33, pages 27842797. Curran Associates, Inc., 2020. 2   
single (robust) classifier. In ArXiv preprint arXiv:1906.09453, 2019. 5   
[ atPR2SatWAUSJu-1020 8213. IEEE, 2020. 8   
[67] Kim Seonghyeon. Implementation of generating diverse high-fidelity images with vq-vae-2 in pytorch, 2020.8   
tion. In Conference on Neural Information Processing Systems (NeurIPS), December 2019. 2   
Vu, Raaeoalcsusealo Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.8   
[r a Workshop and Conference Proceedings, pages 17471756. JMLR.org, 2016. 2   
a  o generation with pixelcnn decoders, 2016. 2, 4, 14   
[Aaron van den Oor, Oiol Vinyals, and Koray Kavkcuoglu. Neural discrete repreentatn learnng, 2018. 3, 4,11   
[ uguksy  oJ ovi   
Aoueaeal sl eual Processing Systems, NeurIPS, 2017. 1, 2, 3   
2018.7   
[kozv   
models, 2021.8 CoRR, abs/1905.10485, 2019. 2   
learning with humans in the loop. arXiv preprint arXiv: 1506.03365, 2015. 5   
[    B Translation. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, 2020. 2   
a Perceptual Metric. In CVPR, 2018. 4, 11, 13   
peptalm. Iri er  atRecin, PR01   
the ade20k dataset. arXiv preprint arXiv:1608.05442, 2016. 6   
  
2019.2