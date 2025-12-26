# 1. Bibliographic Information

## 1.1. Title
One-Minute Video Generation with Test-Time Training

## 1.2. Authors
Karan Dalal, Daniel Koceja, Gashon Hussein, Jiarui Xu, Yue Zhao, Youjin Song, Shihao Han, Ka Chun Cheung, Jan Kautz, Carlos Guestrin, Tatsunori Hashimoto, Sanmi Koyejo, Yejin Choi, Yu Sun, and Xiaolong Wang.

The authors are affiliated with several prominent academic institutions and industrial research labs, including NVIDIA, Stanford University, University of California San Diego (UCSD), University of California Berkeley, and the University of Texas at Austin. This blend of top-tier academic and industry researchers suggests a strong foundation in both theoretical innovation and practical, large-scale implementation.

## 1.3. Journal/Conference
The paper is available as a preprint on arXiv. The abstract indicates a publication date of April 7, 2025, and a note on authorship mentions a submission to the CVPR conference. arXiv is a common platform for researchers to share their work before or during the peer-review process for major conferences. The potential target of CVPR (Conference on Computer Vision and Pattern Recognition) is one of the top-tier venues in the field of computer vision, indicating the high ambition and relevance of this research.

## 1.4. Publication Year
The preprint was submitted in 2025.

## 1.5. Abstract
The paper addresses the challenge of generating long-form (one-minute) videos, a task where standard Transformer models falter due to the quadratic complexity of self-attention. While more efficient alternatives like Mamba exist, their less expressive hidden states struggle with complex, multi-scene narratives. The authors propose using **Test-Time Training (TTT) layers**, a novel type of recurrent layer where the hidden states are themselves neural networks, allowing for greater expressiveness. By integrating TTT layers into a pre-trained 5B parameter Transformer (CogVideo-X) and fine-tuning it, they enable the model to generate one-minute videos from text storyboards. To demonstrate this capability, they created a specialized dataset from "Tom and Jerry" cartoons. In human evaluations, their TTT-based approach significantly outperformed baselines like Mamba 2 and sliding-window attention by 34 Elo points in generating coherent and complex stories. The authors acknowledge that the results still have artifacts, likely due to the base model's limitations, and that their implementation's efficiency could be improved. They conclude that the approach is promising and can be extended to even longer videos.

## 1.6. Original Source Link
- **Official Source Link:** https://arxiv.org/abs/2504.05298
- **PDF Link:** https://arxiv.org/pdf/2504.05298v1.pdf
- **Project Page:** https://test-time-training.github.io/video-dit

  The paper is a preprint and has not yet been officially published in a peer-reviewed venue as of the time of this analysis.

# 2. Executive Summary

## 2.1. Background & Motivation
State-of-the-art video generation models are capable of producing visually stunning, short video clips (typically under 20 seconds). However, they struggle to generate long-form videos, such as those lasting a minute or more, that maintain narrative coherence and tell complex, multi-scene stories.

The core technical hurdle is the **long-context problem**. The dominant architecture, the **Transformer**, relies on a mechanism called `self-attention` which compares every token in a sequence to every other token. This leads to computational and memory costs that grow quadratically with the sequence length ($O(N^2)$). For high-resolution video, a one-minute clip can translate to hundreds of thousands of tokens, making global self-attention computationally prohibitive.

Recent alternatives like **State Space Models (SSMs)**, including `Mamba`, have emerged to address this by offering linear-time complexity ($O(N)$). They function like Recurrent Neural Networks (RNNs), compressing past information into a fixed-size hidden state. However, the paper argues that this compression becomes a bottleneck; a simple matrix-based hidden state is not expressive enough to capture the intricate, long-range dependencies required for a complex story spanning multiple scenes.

This paper's entry point is to bridge this gap between efficiency and expressiveness. It introduces **Test-Time Training (TTT) layers**, a type of recurrent layer where the hidden state is not a simple vector or matrix, but an entire neural network (e.g., a Multi-Layer Perceptron). This "model-as-a-hidden-state" is updated on-the-fly during inference, allowing it to store and process a much richer representation of the video's history.

## 2.2. Main Contributions / Findings
The paper's main contributions are:

1.  **Novel Application of TTT for Long Video Generation:** It is the first work to experiment with Test-Time Training (TTT) layers for the task of long-form video generation. The core innovation is using a neural network as an expressive, updatable hidden state to maintain long-range temporal consistency.

2.  **A Hybrid Architecture for Efficient Long-Context Modeling:** The authors propose a pragmatic architecture that combines the strengths of different approaches. It uses computationally expensive `self-attention` only locally within short 3-second segments, while the efficient `TTT layers` operate globally across the entire one-minute sequence to model the overarching narrative.

3.  **A Proof-of-Concept on a Challenging, Niche Dataset:** To focus specifically on the problem of narrative complexity and dynamic motion, the authors curated a new dataset from ~7 hours of "Tom and Jerry" cartoons. This dataset, with its multi-scene stories and rapid action, serves as an effective testbed for long-range coherence.

4.  **Superior Performance in Human Evaluation:** The proposed method, `TTT-MLP`, significantly outperformed strong, efficient baselines like `Mamba 2` and `Gated DeltaNet`. In a pairwise human comparison, it achieved a 34-point higher Elo score, a meaningful lead that demonstrates a clear preference for the coherence and narrative quality of the videos it generated.

    The key finding is that the expressiveness of the recurrent hidden state is crucial for generating complex, long-form videos. While efficient models like Mamba are a step forward, the TTT approach suggests that investing more computational capacity into the hidden state itself pays significant dividends in temporal consistency and storytelling.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Transformers and Self-Attention
The **Transformer** is a neural network architecture that has become dominant in natural language processing and, more recently, in computer vision. Its key innovation is the **self-attention mechanism**.

-   **Conceptual Definition:** For each element (e.g., a word in a sentence or a patch in an image) in a sequence, self-attention calculates an "attention score" to every other element. These scores determine how much "focus" or influence each element should have on the current one when creating its updated representation. This allows the model to directly capture long-range dependencies.
-   **Mathematical Formula:** The most common form is Scaled Dot-Product Attention. For a set of input vectors, we create three matrices: Query ($Q$), Key ($K$), and Value ($V$). The attention output is calculated as:
    \$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    \$
-   **Symbol Explanation:**
    *   $Q$: The Query matrix, representing the current elements seeking information.
    *   $K$: The Key matrix, representing the elements that can provide information.
    *   $V$: The Value matrix, containing the actual information to be aggregated.
    *   $d_k$: The dimension of the key vectors. The scaling factor $\sqrt{d_k}$ is used to stabilize gradients.
    *   `softmax`: A function that converts scores into a probability distribution, ensuring the weights sum to 1.
-   **Challenge:** The calculation of $QK^T$ involves a matrix multiplication between a sequence of length $N$ and its transpose, resulting in an $N \times N$ attention matrix. This is the source of the quadratic ($O(N^2)$) complexity that makes Transformers inefficient for very long sequences like one-minute videos.

### 3.1.2. Recurrent Neural Networks (RNNs) and State Space Models (SSMs)
**RNNs** are designed to process sequential data. They maintain a **hidden state** that is updated at each timestep, carrying information from the past.

-   **Conceptual Definition:** An RNN processes a sequence one element at a time. At step $t$, it takes the current input $x_t$ and the previous hidden state $h_{t-1}$ to produce a new hidden state $h_t$ and an output $y_t$. This recurrent nature allows it to theoretically handle sequences of any length with linear ($O(N)$) complexity.
-   **State Space Models (SSMs)**, like `Mamba`, are a modern class of models that can be interpreted as a type of linear RNN. They are defined by a state equation that describes how the hidden state `h(t)` evolves and an output equation. They have proven highly effective and efficient for long-sequence modeling. The hidden state in these models is typically a fixed-size vector or matrix.

### 3.1.3. Diffusion Models
**Diffusion Models** are generative models that learn to create data by reversing a noise-adding process.

-   **Conceptual Definition:** The process has two parts:
    1.  **Forward Process:** You start with a real data sample (e.g., an image) and gradually add a small amount of Gaussian noise over many steps, until it becomes pure noise.
    2.  **Reverse Process:** A neural network is trained to reverse this process. Given a noisy image, it learns to predict the noise that was added. By repeatedly subtracting the predicted noise, it can gradually denoise a random noise sample into a realistic new data sample.
-   This paper uses a **Diffusion Transformer (DiT)**, where the denoising network is a Transformer.

### 3.1.4. Test-Time Training (TTT)
**Test-Time Training** is a paradigm where a model's parameters are updated using the test data itself, typically via a self-supervised task.

-   **Conceptual Definition:** Unlike traditional machine learning where a model is trained and then fixed at test time, TTT allows for adaptation to each new test sample. The paper applies this concept to RNNs. Instead of having a fixed hidden state, the hidden state is a neural network model $f$ with weights $W$. As the RNN processes a sequence, it "trains" this inner model $f$ on the incoming tokens. The updated weights $W_t$ serve as the new hidden state. This makes the hidden state far more expressive than a simple vector.

## 3.2. Previous Works
The paper situates its work in the context of several lines of research:

-   **State-of-the-Art Video Generators:** Models like OpenAI's `Sora`, Meta's `MovieGen`, Google's `Veo 2`, and Luma's `Ray 2` have demonstrated incredible realism but are limited to short clips (8-20 seconds) and lack capabilities for complex, multi-scene story generation.
-   **Efficient Transformer Alternatives:** To overcome the quadratic complexity of self-attention, researchers have explored linear-time architectures. This includes linear attention variants and modern RNNs/SSMs like `Mamba` [12] and `DeltaNet` [35, 52]. This paper builds on this trend but criticizes their limited hidden state expressiveness.
-   **Long Video Modeling Techniques:** Previous approaches to extend video length include:
    *   `Sliding-window attention` [3]: Attention is restricted to a local window, which is efficient but loses global context.
    *   `Cascaded models` [15, 50, 55]: A base model generates keyframes, and other models fill in the details or transitions. This is often not end-to-end.
    *   `Story synthesis` [20, 26, 33]: These methods generate a sequence of images or short clips from a story script but often struggle with visual consistency across scenes.
-   **Fast Weight Programmers:** The idea behind TTT layers is inspired by earlier work on "Fast Weight Programmers" [36, 21], where one neural network learns to modify the weights of another network on the fly.

## 3.3. Technological Evolution
The field of video generation has evolved from generating single frames to short, simple clips, and now faces the frontier of long-form, narrative-driven video.
1.  **Early Methods (GANs):** Generated videos frame-by-frame, often suffering from temporal incoherence.
2.  **Short-Clip Transformers/Diffusion:** Models like `Sora` achieved high visual quality and physical realism for single-scene clips by scaling up Transformer and Diffusion architectures.
3.  **Efficiency-Driven Models (SSMs):** Models like `Mamba` were proposed to break the quadratic bottleneck of Transformers, enabling processing of longer sequences in language and showing promise for video.
4.  **Expressiveness-Driven Models (This paper):** This work represents the next step, arguing that linear efficiency is not enough. For complex narratives, the model needs a more powerful way to remember and reason about past events, which they propose to solve with the expressive hidden states of TTT layers.

## 3.4. Differentiation Analysis
Compared to the main related works, this paper's approach is different in the following ways:

-   **vs. Standard Transformers (`Sora`, `Veo`):** It avoids the prohibitive cost of global self-attention by using a hybrid `local attention + global TTT` architecture, making it scalable to minute-long videos.
-   **vs. Efficient RNNs (`Mamba`, `Gated DeltaNet`):** The core innovation lies in the **hidden state**. While Mamba uses a fixed-size matrix, TTT uses an entire MLP. This "model-as-a-hidden-state" can theoretically store and manipulate much more complex information about the sequence history, which is critical for maintaining story coherence.
-   **vs. Sliding Window Attention:** Sliding window attention has a hard limit on its context range. The TTT layer, being recurrent, can theoretically pass information from the very beginning of the sequence to the end, enabling true long-range dependency modeling.

# 4. Methodology

The core of the paper's methodology is the introduction and integration of **Test-Time Training (TTT) layers** into a pre-trained Diffusion Transformer to enable long video generation.

## 4.1. Principles
The fundamental idea is to design a recurrent layer with a highly expressive hidden state. Traditional RNNs struggle to compress long histories into a small, fixed-size vector. The authors draw inspiration from self-supervised learning, where a massive dataset can be compressed into the weights of a neural network.

By analogy, they propose making the RNN's hidden state itself a neural network. As the RNN processes a sequence token by token, it continuously trains this "inner" neural network on the tokens it sees. The updated weights of this inner network serve as the new hidden state. Because a neural network has far greater capacity than a simple vector or matrix, it can store a much richer summary of the history. This process of updating the hidden state via training-like steps, even during inference, gives the layer its name: **Test-Time Training (TTT) Layer**.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. TTT as a Recurrent Layer
A TTT layer processes a sequence of tokens $x_1, \ldots, x_T$ to produce an output sequence $z_1, \ldots, z_T$. Its hidden state at time $t$ is the set of weights $W_t$ of an inner model $f$.

The following diagram from the paper (Figure 2) illustrates the update process.

![该图像是一个示意图，展示了 Test-Time Training (TTT) 层的输入、输出及隐藏状态的更新过程。图中表示了输入标记 $x_t$、输出标记 $z_t$ 以及隐藏状态 $W_t$ 之间的关系。输出规则为 `z_t = f(x_t; W_t)`，更新规则为 `W_t = W_{t-1} - \\eta \\nabla \\ell(W_{t-1}; x_t)`。该结构强调了如何通过动态更新隐藏状态来提升模型表达能力。](images/3.jpg)
*该图像是一个示意图，展示了 Test-Time Training (TTT) 层的输入、输出及隐藏状态的更新过程。图中表示了输入标记 $x_t$、输出标记 $z_t$ 以及隐藏状态 $W_t$ 之间的关系。输出规则为 `z_t = f(x_t; W_t)`，更新规则为 `W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)`。该结构强调了如何通过动态更新隐藏状态来提升模型表达能力。*

The process for generating the output $z_t$ from an input $x_t$ involves two key rules: an update rule and an output rule.

1.  **Update Rule:** The hidden state $W_t$ is updated from the previous state $W_{t-1}$ by performing one step of gradient descent on a self-supervised loss function $\ell$. This loss is computed using the current input token $x_t$.
    \$
    W _ { t } = W _ { t - 1 } - \eta \nabla \ell ( W _ { t - 1 } ; x _ { t } )
    \$
    -   **Symbol Explanation:**
        -   $W_t$: The hidden state at time $t$, which are the weights of the inner model $f$.
        -   $W_{t-1}$: The hidden state from the previous time step.
        -   $\eta$: The inner-loop learning rate, a hyperparameter.
        -   $\nabla \ell ( W _ { t - 1 } ; x _ { t } )$: The gradient of the loss function $\ell$ with respect to the weights $W_{t-1}$, computed using the current input token $x_t$.

2.  **Output Rule:** The output token $z_t$ is generated by passing the input token $x_t$ through the inner model $f$ using the *newly updated* weights $W_t$.
    \$
    z _ { t } = f ( x _ { t } ; W _ { t } )
    \$
    -   **Symbol Explanation:**
        -   $z_t$: The output of the TTT layer at time $t$.
        -   $f(x_t; W_t)$: The forward pass of the inner model $f$ on input $x_t$ with parameters $W_t$.

### 4.2.2. Learning the Self-Supervised Task
The effectiveness of TTT depends on the self-supervised task defined by the loss $\ell$. A simple choice is reconstruction, where the model tries to reconstruct the input $x_t$ from a corrupted version $\tilde{x}_t$.
\$
\ell ( W ; x _ { t } ) = \| f ( \tilde { x } _ { t } ; W ) - x _ { t } \| ^ { 2 }
\$
Instead of handcrafting this task, the authors make it learnable. Inspired by the Query-Key-Value mechanism in self-attention, they introduce three learnable projection matrices: $\theta_Q, \theta_K, \theta_V$. These matrices are part of the outer model and are learned during the main training (fine-tuning) phase.

1.  **Learnable Loss Function:** The input to the inner model $f$ is a projection of $x_t$ (the "key"), and the reconstruction target is another projection (the "value").
    \$
    \ell ( W ; x _ { t } ) = \| f ( \theta _ { K } x _ { t } ; W ) - \theta _ { V } x _ { t } \| ^ { 2 }
    \$
    -   **Symbol Explanation:**
        -   $\theta_K$: A learnable matrix that projects $x_t$ into a "key" representation. This is the input to the inner model for the self-supervised task.
        -   $\theta_V$: A learnable matrix that projects $x_t$ into a "value" representation. This is the reconstruction target.

2.  **Learnable Output Rule:** The final output $z_t$ is computed using a third projection of the input (the "query").
    \$
    z _ { t } = f \left( \theta _ { Q } x _ { t } ; W _ { t } \right)
    \$
    -   **Symbol Explanation:**
        -   $\theta_Q$: A learnable matrix that projects $x_t$ into a "query" representation, which is the input for the final output computation.

### 4.2.3. TTT-MLP Instantiation
The paper instantiates the inner model $f$ as a two-layer Multi-Layer Perceptron (MLP) with a residual connection and Layer Normalization for stability. This specific instantiation is called `TTT-MLP`.
\$
f ( x ) = x + \mathsf{ L N } ( f _ { \mathsf { M L P } } ( x ) )
\$
-   **Symbol Explanation:**
    -   $f_{\mathsf{MLP}}(x)$: A standard two-layer MLP with a GELU activation function and a hidden dimension $4\times$ the input dimension.
    -   $\mathsf{LN}$: Layer Normalization.

### 4.2.4. Architectural Integration and Pipeline
The authors integrate `TTT-MLP` layers into a pre-trained `CogVideo-X` 5B Diffusion Transformer.

The diagram below (Figure 4 from the paper) shows the overall architecture and data flow.

![该图像是示意图，展示了包含 TTT 层的 Transformer 模型架构及其在生成一段一分钟视频时的处理过程。左侧展示了模型的内部结构，包括 Gate、TTT Layer、Local Attention 和 LayerNorm，右侧显示了一分钟视频的分段和对应的文本描述。](images/4.jpg)
*该图像是示意图，展示了包含 TTT 层的 Transformer 模型架构及其在生成一段一分钟视频时的处理过程。左侧展示了模型的内部结构，包括 Gate、TTT Layer、Local Attention 和 LayerNorm，右侧显示了一分钟视频的分段和对应的文本描述。*

1.  **Gating Mechanism:** To smoothly introduce the TTT layers (which are initialized from scratch) into the powerful pre-trained model without disrupting it, they use a gating mechanism. The output of the TTT block is blended with the original input sequence via a learned gate vector $\alpha$.
    \$
    \mathtt { g a t e } ( \mathsf { T T T } , X ; \alpha ) = \operatorname { t a n h } ( \alpha ) \otimes \mathsf { T T T } ( X ) + X
    \$
    -   **Symbol Explanation:**
        -   $X$: The input sequence to the block.
        -   $\mathsf{TTT}(X)$: The output sequence from the TTT layer.
        -   $\alpha$: A learnable vector, initialized near zero, so that initially the TTT layer's contribution is minimal.
        -   $\otimes$: Element-wise multiplication.

2.  **Bi-directional Processing:** Diffusion models are non-causal (they can see the whole sequence). To allow the TTT layer (which is causal) to process information from both past and future, they apply it twice: once in the forward direction and once in the reverse direction. The final modified Transformer block replaces the standard residual connection with this bi-directional TTT processing.

3.  **Local Attention, Global TTT:** This is a key design choice for efficiency. For a one-minute video broken into multiple 3-second segments:
    -   **Self-attention layers** operate **locally** within each 3-second segment. This keeps their quadratic cost manageable.
    -   **TTT layers** operate **globally** across the entire concatenated sequence of tokens for the full minute. This allows them to capture long-range dependencies across scenes, leveraging their linear-time complexity.

### 4.2.5. Parallelization for Training and Inference
The recurrent nature of the TTT update rule (computing $W_t$ requires $W_{t-1}$) is inherently sequential. To make it efficient on GPUs, they parallelize the computation over small mini-batches of $b$ tokens.

1.  **Inner-loop Mini-batch Update:** Instead of updating the weights token by token, they compute gradients for a batch of $b$ tokens using the same starting weights $W_{(i-1)b}$ and then average these gradients to perform a single, more stable update.
    \$
    { \cal W } _ { i b } = { \cal W } _ { ( i - 1 ) b } - \frac { \eta } { b } \sum _ { t = ( i - 1 ) b + 1 } ^ { i b } \nabla \ell \bigl ( W _ { ( i - 1 ) b } ; x _ { t } \bigr )
    \$
2.  **Inner-loop Mini-batch Output:** The outputs for all $b$ tokens in the mini-batch are then computed in parallel using the single new set of weights $W_{ib}$.
    \$
    z _ { t } = f ( W _ { i b } ; x _ { t } ) , \qquad \mathrm { f o r } \ t = ( i - 1 ) b + 1 , \dots , i b
    \$

### 4.2.6. On-Chip Tensor Parallelism
The hidden state of `TTT-MLP` (the weights of the two-layer MLP) is too large to fit in the fast on-chip memory (SMEM) of a single GPU Streaming Multiprocessor (SM). To overcome this, they implement a custom kernel that uses **Tensor Parallelism** across the SMs of a single GPU. The MLP weights are sharded across multiple SMs, and computations like AllReduce are performed directly between SMs using fast on-chip interconnects. This minimizes slow data transfers to and from the main GPU memory (HBM), significantly improving efficiency.

# 5. Experimental Setup

## 5.1. Datasets
The authors curated a custom dataset specifically for this work, as existing datasets did not emphasize long-range, multi-scene narrative complexity.

-   **Source:** 81 episodes of the classic "Tom and Jerry" cartoons from 1940-1948, totaling approximately 7 hours of video.
-   **Preprocessing:** The original low-resolution videos were upscaled using a super-resolution model to a standard $720 \times 480$ resolution.
-   **Annotation:** Human annotators performed a two-step process:
    1.  They segmented each episode into distinct scenes.
    2.  They wrote detailed, paragraph-long descriptions for each 3-second segment of video, forming a storyboard.
-   **Multi-stage Training Data:** To gradually train the model on longer contexts, they created datasets of varying lengths (3, 9, 18, 30, and 63 seconds) by concatenating contiguous segments and their corresponding text annotations.

    The following figure (from the paper's appendix) shows examples of the different text prompt formats used, from a high-level summary to a detailed storyboard.

    ![该图像是一个插图，展示了《汤姆和杰瑞》中的多个场景，体现了故事情节的连贯性和动态变化。这些场景展示了汤姆和杰瑞之间的经典追逐，反映了复杂的多场景叙事。为了验证生成视频的效果，作者使用了基于此类卡通的数据集进行实验。](images/8.jpg)
    *该图像是一个插图，展示了《汤姆和杰瑞》中的多个场景，体现了故事情节的连贯性和动态变化。这些场景展示了汤姆和杰瑞之间的经典追逐，反映了复杂的多场景叙事。为了验证生成视频的效果，作者使用了基于此类卡通的数据集进行实验。*

This dataset choice is strategic: it is rich in dynamic motion and complex, multi-scene interactions, allowing the evaluation to focus on temporal coherence and storytelling rather than photorealism.

## 5.2. Evaluation Metrics
The primary evaluation was conducted through human studies, as automated metrics for long-form video quality and narrative coherence are still unreliable.

-   **Methodology:** The evaluation used pairwise, blind comparisons. Human evaluators were shown two videos generated by different methods from the same text prompt and asked to choose which one was better along a randomly selected axis.
-   **Evaluation Axes:** Four axes were adapted from the MovieGen benchmark:
    1.  **Text following:** How well the video aligns with the provided text prompt.
    2.  **Motion naturalness:** The realism of character movements and physics.
    3.  **Aesthetics:** The visual appeal, including lighting, color, and composition.
    4.  **Temporal consistency:** The coherence of objects, characters, and scenes over time, both within and across scenes.

-   **Core Metric: Elo Rating System**
    -   **Conceptual Definition:** The Elo rating system is a method for calculating the relative skill levels of players in zero-sum games (like chess). In this context, each video generation method is a "player." When a method's video is preferred over another's in a pairwise comparison, its Elo score increases, and the loser's score decreases. It provides a continuous, relative ranking of all methods. A higher Elo score indicates better performance.
    -   **Mathematical Formula:** The expected score of Player A against Player B is given by: $E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$. The new rating for Player A after a match is: `R'_A = R_A + K(S_A - E_A)`.
    -   **Symbol Explanation:**
        *   $R_A, R_B$: The current ratings of players A and B.
        *   $E_A$: The expected score of player A (the probability of winning).
        *   $S_A$: The actual score of the match (1 for a win, 0 for a loss, 0.5 for a draw).
        *   $K$: A constant that determines how much the rating is updated after each game.
    -   The paper uses the implementation from `LMSys Chatbot Arena`.

## 5.3. Baselines
The proposed `TTT-MLP` model was compared against several strong baselines that also have linear complexity, making for a fair comparison of efficiency and quality. All baselines were integrated into the same `CogVideo-X` backbone and fine-tuned using the same recipe.

-   **Local attention:** The original `CogVideo-X` model without any global context mechanism. It processes each 3-second segment independently.
-   **TTT-Linear:** A variant of the proposed method where the inner model $f$ is a simple linear layer instead of an MLP. This tests the importance of the hidden state's non-linearity.
-   **Mamba 2:** A state-of-the-art State Space Model known for its efficiency and strong performance on language tasks. It uses a matrix-based hidden state.
-   **Gated DeltaNet:** An advanced RNN based on the delta rule, which improves upon `Mamba 2` with a more sophisticated update mechanism.
-   **Sliding-window attention (SWA):** A common baseline for long-sequence Transformers where self-attention is computed only within a fixed-size local window of tokens.

# 6. Results & Analysis

## 6.1. Core Results Analysis (63-Second Videos)
The main results are from the human evaluation of one-minute (63-second) videos. The following are the results from Table 1 of the original paper:

| | Text following | Motion naturalness | Aesthetics | Temporal consistency | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Mamba 2 | 985 | 976 | 963 | 988 | 978 |
| Gated DeltaNet | 983 | 984 | 993 | 1004 | 991 |
| Sliding window | 1016 | 1000 | 1006 | 975 | 999 |
| **TTT-MLP** | **1014** | **1039** | **1037** | **1042** | **1033** |

-   **Overall Performance:** `TTT-MLP` is the clear winner, achieving an average Elo score of 1033. This is **34 points higher** than the next best method, sliding window attention (999). The paper provides context for this margin by noting that GPT-4o's lead over GPT-4 Turbo in the LMSys Chatbot Arena is 29 Elo points, suggesting the 34-point gap is highly significant and represents a clear human preference.
-   **Axis-specific Strengths:** `TTT-MLP` shows its largest leads in `Motion naturalness` (+39 points over the best baseline on this axis) and `Temporal consistency` (+38 points). This directly validates the central hypothesis of the paper: the more expressive hidden state of TTT layers is particularly effective at maintaining coherence and realistic dynamics over long durations.
-   **Baseline Comparison:** While `Gated DeltaNet` outperforms `Mamba 2`, both fall significantly behind `TTT-MLP`, supporting the argument that their matrix-based hidden states are a bottleneck for complex story generation. Sliding window attention performs surprisingly well but is ultimately limited in true long-range reasoning, as shown by its lower score on `Temporal consistency`.

    The following images (from Figure 5 of the paper) provide a qualitative comparison, visually showing the superior coherence of videos generated by `TTT-MLP`.

    ![该图像是一组经典的《猫和老鼠》动画截图，展示了汤姆猫和杰瑞鼠的互动场景。每一帧都表现出它们之间的幽默追逐，呈现了多样的场景和情感，体现了动画生动的叙事风格。](images/6.jpg)
    *该图像是一组经典的《猫和老鼠》动画截图，展示了汤姆猫和杰瑞鼠的互动场景。每一帧都表现出它们之间的幽默追逐，呈现了多样的场景和情感，体现了动画生动的叙事风格。*

    ![该图像是插图，展示了《汤姆与杰瑞》中一系列经典场景，表现了汤姆猫与杰瑞鼠之间的互动与追逐。每个画面都呈现出他们在不同情境中的幽默表现，体现了多场景故事叙述的特点。](images/7.jpg)
    *该图像是插图，展示了《汤姆与杰瑞》中一系列经典场景，表现了汤姆猫与杰瑞鼠之间的互动与追逐。每个画面都呈现出他们在不同情境中的幽默表现，体现了多场景故事叙述的特点。*

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. 18-Second Elimination Round
To reduce evaluation costs, an initial round was conducted on shorter, 18-second videos. The following are the results from Table 3 of the original paper's appendix:

| | Text following | Motion naturalness | Aesthetics | Temporal consistency | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Local Attention | 965 | 972 | 969 | 944 | 962 |
| TTT-Linear | 1003 | 995 | 1007 | 1001 | 1001 |
| Mamba 2 | 1023 | 987 | 1008 | 1004 | 1005 |
| **Gated DeltaNet** | **1020** | **1039** | **1044** | **1026** | **1032** |
| SWA | 995 | 1004 | 993 | 980 | 993 |
| TTT-MLP | 994 | 1002 | 1002 | 1019 | 1004 |

-   **Key Finding:** On these shorter videos (approx. 100k tokens), **`Gated DeltaNet` performed the best**, outperforming `TTT-MLP` by 28 Elo points. `TTT-MLP`'s performance was comparable to `Mamba 2` and `TTT-Linear`.
-   **Analysis:** This result is crucial as it highlights a trade-off. For moderately long contexts, the simpler and more efficient matrix hidden states of models like `Gated DeltaNet` are highly effective and perhaps easier to optimize. The greater expressive power of `TTT-MLP`'s complex hidden state seems to provide its main advantage only when the context becomes much longer (e.g., 63 seconds, >300k tokens), where the limitations of simpler hidden states become apparent.

### 6.2.2. Efficiency Analysis
The paper analyzes the wall-clock time for inference and training. The following charts (Figure 6) compare the different methods.

![Figure 6. For 63-second videos, inference with full attention (over 300k tokens) would have taken $1 1 \\times$ longer than local attention, and training $1 2 \\times$ longer, as discussed in Section 1. TTT-MLP takes $2 . 5 \\times$ and $3 . 8 \\times$ respectively significantly more efficient than full attention, but still less efficient than, for example, Gated DeltaNet, which takes $1 . 8 \\times$ longer than local attention in both inference and training.](images/9.jpg)
*该图像是一个图表，展示了不同视频长度下的推理和训练延迟时间。左侧图表显示推理延迟，右侧则展示训练延迟，比较了全注意力、TTT-MLP、Gated DeltaNet、Mamba 2和局部注意力的方法。可以看出，TTT-MLP在推理和训练中的性能优于全注意力，但仍然逊色于Gated DeltaNet.*

-   **Inference and Training Latency:** `TTT-MLP` is significantly more efficient than hypothetical full self-attention (2.5x faster in inference, 3.8x faster in training for 63s videos). However, it is still noticeably slower than the more streamlined RNN baselines. For example, `Gated DeltaNet` is 1.8x slower than local attention, while `TTT-MLP` is 2.5x slower.
-   **Conclusion:** There is an explicit trade-off between expressiveness and efficiency. The `TTT-MLP`'s complex hidden state and inner-loop gradient computations incur a higher computational cost than the highly optimized operations of `Mamba 2` and `Gated DeltaNet`.

## 6.3. Limitations and Artifacts
The authors are transparent about the limitations of their current work.

-   **Video Artifacts:** The generated videos, while coherent, still contain visual flaws, particularly in motion and aesthetics. The paper shows examples of these artifacts in Figure 7. They speculate that these issues may stem from the limited capabilities of the base `CogVideo-X 5B` model rather than the TTT approach itself.
    -   ![该图像是插图，展示了《汤姆与杰瑞》中汤姆猫和杰瑞鼠的互动场景。画面分为四个格子，分别描绘了他们在进行各种搞笑行为，比如搬箱子和玩弄机关，这些动作反映出他们典型的对立和幽默风格。](images/10.jpg) (Temporal consistency issues within scenes)
        *该图像是插图，展示了《汤姆与杰瑞》中汤姆猫和杰瑞鼠的互动场景。画面分为四个格子，分别描绘了他们在进行各种搞笑行为，比如搬箱子和玩弄机关，这些动作反映出他们典型的对立和幽默风格。*
    -   ![该图像是一个动画片段，展示了汤姆与杰瑞的一幕，其中杰瑞在奔跑并躲避从上方掉落的奶酪块。此图像包含了多个帧，展示了动作的流动性和动态效果。](images/11.jpg) (Unnatural motion/physics)
        *该图像是一个动画片段，展示了汤姆与杰瑞的一幕，其中杰瑞在奔跑并躲避从上方掉落的奶酪块。此图像包含了多个帧，展示了动作的流动性和动态效果。*
    -   ![该图像是来自《猫和老鼠》的插图，展示了汤姆猫在追逐一只老鼠的过程。每一帧展示了汤姆猫的动作和表情，形象生动，突显了卡通中的幽默与紧张情节。](images/12.jpg) (Inconsistent aesthetics/lighting)
        *该图像是来自《猫和老鼠》的插图，展示了汤姆猫在追逐一只老鼠的过程。每一帧展示了汤姆猫的动作和表情，形象生动，突显了卡通中的幽默与紧张情节。*
-   **Wall-clock Time:** As discussed, `TTT-MLP` is less efficient than other linear-time baselines, presenting an area for future optimization.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that generating coherent, minute-long videos with complex narratives is feasible by enhancing a pre-trained Transformer with **Test-Time Training (TTT) layers**. The core contribution is the use of a neural network (an MLP) as a highly expressive recurrent hidden state, which is updated on-the-fly. This `TTT-MLP` approach, combined with a pragmatic `local attention, global TTT` architecture, proved significantly superior to strong baselines like `Mamba 2` and `Gated DeltaNet` in human evaluations, particularly on measures of temporal consistency and motion quality for long videos. The work serves as a strong proof-of-concept that the expressiveness of the hidden state is a critical factor for long-form generative modeling.

## 7.2. Limitations & Future Work
The authors identified several limitations and corresponding directions for future research:

-   **Limitations:**
    1.  **Video Quality:** Generated videos still contain noticeable artifacts, likely inherited from the base model.
    2.  **Efficiency:** The `TTT-MLP` implementation is slower than other modern RNNs like `Gated DeltaNet`.
    3.  **Short-Context Performance:** TTT-MLP was not the top performer on shorter (18-second) videos, where simpler models excelled.

-   **Future Work:**
    1.  **Faster Implementation:** Optimizing the TTT-MLP kernel further by managing GPU resources like registers more effectively.
    2.  **Better Integration:** Exploring more sophisticated ways to integrate TTT layers into pre-trained models beyond simple gating and bi-direction.
    3.  **Longer Videos & Larger Hidden States:** Scaling the approach to generate even longer videos by using larger neural networks (e.g., a small Transformer) as the hidden state, further pushing the boundaries of expressiveness.

## 7.3. Personal Insights & Critique
This paper offers several valuable insights and opens up exciting research avenues.

-   **Model-as-Hidden-State is a Powerful Concept:** The core idea of using a trainable model as the hidden state is extremely powerful and feels like a natural evolution for recurrent architectures. It elegantly reframes the problem of information compression from "squeezing vectors" to "learning a summary model." This concept could be widely applicable beyond video generation to any domain requiring deep, long-range reasoning, such as document-level language modeling or reinforcement learning.

-   **Pragmatism in Hybrid Architectures:** The `local attention, global TTT` design is a clever and practical engineering choice. It acknowledges that no single architecture is perfect and instead combines the local pattern-matching strength of attention with the global, linear-time reasoning of a powerful RNN. This hybrid approach is a promising path forward for building scalable models.

-   **The Importance of Focused Datasets:** The strategic choice to create a "Tom and Jerry" dataset was brilliant for a proof-of-concept. It allowed the researchers to isolate and make progress on the core challenge of narrative coherence and dynamic motion, without being confounded by the immense difficulty of achieving photorealism.

-   **Critique and Open Questions:**
    *   **Generalization:** The most significant question is how well this approach generalizes beyond cartoon animations to open-domain, photorealistic video. The complexities of real-world physics, human appearance, and environmental consistency are of a different magnitude.
    *   **Scalability of Inner-Loop Training:** While the hidden state is more expressive, it's also much more computationally intensive to update. The paper uses a single gradient step. More complex inner models might require more sophisticated or multi-step optimization, further increasing the cost. The trade-off between expressiveness and computational feasibility will remain a central challenge.
    *   **Dependence on Pre-trained Models:** The approach relies on fine-tuning a powerful base model. This means its ultimate quality is capped by the base model, and the cost of pre-training such a model remains a barrier for most researchers. A method that works well from scratch would be even more impactful.

        Overall, this paper presents a compelling and creative solution to a major bottleneck in generative AI. It convincingly argues for a new direction in recurrent model design and provides strong evidence that investing in more expressive hidden states is key to unlocking long-form, coherent generation.