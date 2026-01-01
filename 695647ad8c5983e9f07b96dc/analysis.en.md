# 1. Bibliographic Information

## 1.1. Title
mHC: Manifold-Constrained Hyper-Connections

The title clearly states the paper's core contribution: a modification of the `Hyper-Connections` (HC) architecture by imposing a `manifold constraint`. This suggests a method that retains the benefits of HC while improving its properties, likely related to stability or regularization.

## 1.2. Authors
Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, and Wenfeng Liang.

All authors are affiliated with **DeepSeek-AI**. This indicates that the research is a product of a large, industry-focused research lab known for developing large-scale language models. This background suggests that the work is likely driven by practical challenges encountered during the training of state-of-the-art foundation models.

## 1.3. Journal/Conference
The paper was submitted to arXiv, a preprint server for academic papers. The publication date is listed as December 31, 2025, which is a future placeholder. The arXiv identifier (`2512.24880`) suggests a submission date of December 2025.

As a preprint, this work has not yet undergone formal peer review. However, arXiv is the standard platform for disseminating cutting-edge research in the fields of AI and machine learning, allowing for rapid sharing of results with the community.

## 1.4. Publication Year
The paper is a preprint with a prospective date in late 2025. Given the current date of analysis (January 2026), it is considered very recent work.

## 1.5. Abstract
The abstract introduces `Hyper-Connections (HC)` as a recent extension of the standard `residual connection` paradigm. While HC improves performance by expanding the residual stream's width and connectivity, it suffers from two major drawbacks: **(1) training instability** due to the loss of the `identity mapping` property, and **(2) significant memory access overhead**. To solve these problems, the authors propose **Manifold-Constrained Hyper-Connections (mHC)**. The core idea of mHC is to project the `residual connection` space of HC onto a specific mathematical manifold to restore the `identity mapping` property, thereby ensuring training stability. Additionally, mHC incorporates infrastructure-level optimizations to maintain efficiency. The authors state that empirical results show mHC is effective and scalable for large-model training, offering performance gains. They conclude by positioning mHC as a practical framework that could inspire future work in topological network design.

## 1.6. Original Source Link
- **Original Source Link:** `https://arxiv.org/abs/2512.24880`
- **PDF Link:** `https://arxiv.org/pdf/2512.24880v1.pdf`

  The links point to the preprint version of the paper on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The `residual connection`, introduced by ResNet, has been a cornerstone of deep learning for a decade. It enables the training of very deep neural networks by allowing signals to bypass layers through an `identity mapping`, which prevents the "vanishing gradient" problem and ensures stable signal propagation. This can be expressed as: `output = input + function(input)`.

Recently, architectures like `Hyper-Connections (HC)` have sought to enhance this paradigm by expanding the single "highway" of the residual connection into a multi-lane superhighway. Instead of a single stream of information, HC uses multiple parallel residual streams and introduces learnable matrices to control how information is mixed between these streams and processed by the network layers. While this approach has demonstrated performance gains, it comes at a high cost:

1.  **Loss of Identity Mapping and Instability:** The learnable matrices in HC are unconstrained. When these matrices are applied repeatedly across many layers, they can cause the signal to either explode to infinity or vanish to zero. This breaks the fundamental stability guarantee of the original residual connection, leading to unstable training, especially in very large models.
2.  **System Overhead:** Expanding the residual stream from one lane to $n$ lanes ($n > 1$) increases the amount of data that needs to be read from and written to GPU memory. This creates a memory I/O bottleneck, which slows down training throughput.

    The core motivation of this paper is to **fix the shortcomings of Hyper-Connections without sacrificing their benefits**. The authors want to create a version of HC that is stable, scalable for massive models, and efficient in terms of hardware utilization.

## 2.2. Main Contributions / Findings
The paper introduces **Manifold-Constrained Hyper-Connections (mHC)** and makes the following key contributions:

1.  **Identifies the Root Cause of HC's Instability:** The paper provides a clear theoretical and empirical analysis showing that the instability of HC originates from the unconstrained learnable matrix ($H_l^res$) that governs the residual stream, which disrupts the `identity mapping` property across layers.

2.  **Proposes Manifold-Constrained Hyper-Connections (mHC):** The central innovation is to constrain the problematic residual matrix $H_l^res$ to a specific mathematical space known as a **manifold of doubly stochastic matrices**. A doubly stochastic matrix has all non-negative entries, and every row and column sums to 1. This constraint ensures that the matrix acts as a non-expansive operator, preventing signal explosion and guaranteeing stable propagation, thus restoring a generalized form of the identity mapping.

3.  **Introduces a Practical Projection Method:** The paper employs the well-established **Sinkhorn-Knopp algorithm** to project the learnable matrix onto the doubly stochastic manifold during training.

4.  **Delivers Rigorous Infrastructure Optimization:** Recognizing that architectural changes impact hardware performance, the authors develop a suite of system-level optimizations, including custom `kernel fusion`, a `recomputation` strategy to save memory, and an enhanced pipeline parallelism schedule (`DualPipe`) to hide communication latency. These optimizations make mHC practical for large-scale training with only a marginal `6.7%` time overhead.

5.  **Achieves Superior Stability and Performance:** Through extensive experiments on large language models (up to 27B parameters), the paper demonstrates that mHC resolves the training instability of HC, scales effectively to larger models and more data, and achieves superior performance on various downstream tasks, particularly in reasoning.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Residual Connection (ResNet)
The residual connection is a fundamental architectural innovation proposed in "Deep Residual Learning for Image Recognition" (ResNet). Before ResNet, training very deep neural networks was notoriously difficult due to the **vanishing gradient problem**, where gradients shrink exponentially as they are backpropagated through many layers, making it impossible for earlier layers to learn.

A residual block addresses this by creating a "shortcut" or "skip connection" that allows the gradient to flow directly. The output of a layer $l$ is not just the result of a transformation $\mathcal{F}$, but the sum of the input and the transformation.

The formula for a single residual layer is:
\$
\mathbf{x}_{l+1} = \mathbf{x}_{l} + \mathcal{F}(\mathbf{x}_{l}, \mathcal{W}_{l})
\$
where:
*   $\mathbf{x}_{l}$ is the input to the $l$-th layer.
*   $\mathbf{x}_{l+1}$ is the output of the $l$-th layer.
*   $\mathcal{F}(\mathbf{x}_{l}, \mathcal{W}_{l})$ is the residual function (e.g., a set of convolutional layers or a Transformer block) with learnable weights $\mathcal{W}_{l}$.

    The key term is $\mathbf{x}_{l}$, which forms the **identity mapping**. This ensures that even if the function $\mathcal{F}$ learns nothing (i.e., its output is zero), the output of the block is still the identity of the input. This makes learning easier and allows signals to propagate through hundreds or even thousands of layers without degradation.

### 3.1.2. Transformer Architecture
The Transformer, introduced in "Attention Is All You Need," is the dominant architecture for modern Large Language Models (LLMs). A standard Transformer block, which would represent the function $\mathcal{F}$ in the context of a residual connection, is composed of two main sub-layers:
1.  **Multi-Head Self-Attention (MHSA):** This mechanism allows the model to weigh the importance of different words in the input sequence when processing a specific word. It computes a representation of each token by attending to all other tokens.
2.  **Feed-Forward Network (FFN):** This is a simple, fully connected neural network applied independently to each token's representation. It typically consists of two linear layers with a non-linear activation function in between.

    In modern LLMs, both the attention and FFN sub-layers are wrapped with residual connections and layer normalization.

### 3.1.3. Doubly Stochastic Matrix
A doubly stochastic matrix is a square matrix with real, non-negative entries where each row and each column sums to 1. For an $n \times n$ matrix $A = [a_{ij}]$:
1.  $a_{ij} \geq 0$ for all `i, j`.
2.  $\sum_{j=1}^{n} a_{ij} = 1$ for each row $i$.
3.  $\sum_{i=1}^{n} a_{ij} = 1$ for each column $j$.

**Key Properties for this Paper:**
*   **Norm Preservation:** The spectral norm (the largest singular value) of a doubly stochastic matrix is exactly 1. This means that when you multiply a vector by such a matrix, its length (norm) will not increase. This property is crucial for preventing the "exploding signal" problem.
*   **Compositional Closure:** The product of two or more doubly stochastic matrices is also a doubly stochastic matrix. This ensures that even when these matrices are applied sequentially across many layers, the resulting composite transformation remains stable and non-expansive.
*   **Birkhoff Polytope:** The set of all $n \times n$ doubly stochastic matrices forms a convex polytope known as the Birkhoff polytope. The vertices of this polytope are the permutation matrices. This means any doubly stochastic matrix can be expressed as a convex combination of permutation matrices, giving it a clear geometric interpretation as a "soft" or "mixed" permutation.

## 3.2. Previous Works
The paper positions itself within the field of **macro-design**, which focuses on the high-level topological structure of neural networks.

### 3.2.1. Hyper-Connections (HC)
`Hyper-Connections (HC)` is the direct predecessor to mHC. HC modifies the standard residual connection by expanding the channel dimension $C$ of the residual stream by a factor of $n$. This creates an $n \times C$ dimensional hidden state, which can be seen as $n$ parallel residual streams.

The core formula for a single layer of HC is:
\$
\mathbf{x}_{l+1} = \mathcal{H}_{l}^{\mathrm{res}} \mathbf{x}_{l} + \mathcal{H}_{l}^{\mathrm{post}\top} \mathcal{F}(\mathcal{H}_{l}^{\mathrm{pre}} \mathbf{x}_{l}, \mathcal{W}_{l})
\$
where:
*   $\mathbf{x}_{l}, \mathbf{x}_{l+1} \in \mathbb{R}^{n \times C}$ are the multi-stream input and output.
*   $\mathcal{F}$ is the standard residual function (e.g., a Transformer block) that operates on a $C$-dimensional input.
*   $\mathcal{H}_{l}^{\mathrm{pre}} \in \mathbb{R}^{1 \times n}$ is a learnable vector that **aggregates** information from the $n$ streams before feeding it to $\mathcal{F}$.
*   $\mathcal{H}_{l}^{\mathrm{post}} \in \mathbb{R}^{1 \times n}$ is a learnable vector that **distributes** the output of $\mathcal{F}$ back to the $n$ streams.
*   $\mathcal{H}_{l}^{\mathrm{res}} \in \mathbb{R}^{n \times n}$ is a learnable matrix that **mixes** information among the $n$ residual streams.

    The key issue, as identified by the mHC paper, lies in the recursive application of $\mathcal{H}_{l}^{\mathrm{res}}$. Over `L-l` layers, the original input $\mathbf{x}_l$ is transformed by a product of matrices:
\$
\mathbf{x}_{L} = \left( \prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\mathrm{res}} \right) \mathbf{x}_{l} + \dots
\$
Since the matrices $\mathcal{H}_{i}^{\mathrm{res}}$ are unconstrained, this product can lead to eigenvalues much larger or smaller than 1, causing the signal to explode or vanish, leading to numerical instability.

### 3.2.2. Other Macro-Designs
*   **DenseNet:** Proposed connecting each layer to every other layer in a feed-forward fashion, creating a "dense" connectivity pattern to encourage feature reuse and improve gradient flow.
*   **FractalNet:** Used a recursive fractal structure to create deep networks with many short and long paths for signals to travel, avoiding the need for residual connections altogether.
*   **RMT (Residual Matrix Transformer) & MUDDFormer:** More recent works that also explore expanding the residual stream, but like HC, they introduce unconstrained learn-able connections that can risk instability.

## 3.3. Technological Evolution
1.  **Shallow Networks:** Early neural networks were relatively shallow.
2.  **Deep Networks with Residual Connections (ResNet):** The introduction of residual connections enabled training of networks with hundreds of layers, leading to breakthroughs in computer vision.
3.  **Transformers:** The Transformer architecture, with its inherent residual connections, became the standard for NLP and now for many other domains.
4.  **Wider Residual Streams (HC, etc.):** Recent research began exploring not just depth but the "width" and complexity of the residual connection itself. HC was a key example, showing performance gains but also exposing the fragility of breaking the `identity mapping` property.
5.  **Stabilized Wide Streams (mHC):** This paper represents the next step: reaping the benefits of wider residual streams while re-introducing the stability of the original residual connection through principled mathematical constraints.

## 3.4. Differentiation Analysis
The core innovation of mHC relative to its predecessors is the **principled stabilization of the multi-stream residual connection**.

*   **vs. Standard Residual Connection:** mHC is a generalization. A standard residual connection is equivalent to mHC with an expansion rate $n=1$. mHC allows for richer, learnable interactions between parallel information streams.
*   **vs. Hyper-Connections (HC):** While HC introduced the idea of a multi-stream connection, it did so with unconstrained matrices, leading to instability. mHC is a direct **fix** for HC. It replaces the "anything-goes" learnable matrix $\mathcal{H}^{\mathrm{res}}$ with a constrained doubly stochastic matrix, which is theoretically guaranteed to be stable.
*   **vs. Other Macro-Designs:** Unlike methods like DenseNet which increase connectivity (and computation/memory) by adding more paths, mHC focuses on enriching the existing residual path. The explicit focus on infrastructure optimization also sets it apart, making it a more practical solution for today's massive models.

# 4. Methodology

## 4.1. Principles
The central idea of `mHC` is to preserve the stability of the `identity mapping` from ResNets while still allowing for the complex interactions of multi-stream architectures like `HC`. The original `identity mapping` ensures stability because the transformation matrix for the residual stream is the identity matrix, $\mathbf{I}$, which has a spectral norm of 1.

The authors of mHC generalize this principle. Instead of forcing the residual matrix $\mathcal{H}_l^{\mathrm{res}}$ to be the identity matrix (which would prevent any mixing between streams), they constrain it to belong to the manifold of **doubly stochastic matrices**. This choice is motivated by several key theoretical properties:

1.  **Norm Preservation:** A doubly stochastic matrix has a spectral norm of at most 1, making it non-expansive. This prevents the signal from exploding during the forward pass and the gradient from exploding during the backward pass.
2.  **Compositional Closure:** The product of doubly stochastic matrices is also doubly stochastic. This is critical because it means that even when applied across many layers, the composite transformation $\prod \mathcal{H}_{L-i}^{\mathrm{res}}$ remains non-expansive and stable.
3.  **Convex Combination Interpretation:** A doubly stochastic matrix transforms a vector by taking a convex combination of its elements. This provides a robust mechanism for fusing features across the multiple residual streams without wild fluctuations.

    In essence, mHC replaces the unstable, arbitrary linear transformations of HC with stable, well-behaved convex combinations, thus restoring a generalized and more expressive form of the identity mapping principle.

## 4.2. Core Methodology In-depth (Layer by Layer)
The mHC method can be broken down into two main parts: (1) calculating the unconstrained mapping coefficients and (2) projecting them onto the desired manifolds.

### 4.2.1. Parameterization of Unconstrained Mappings
The process begins at each layer $l$ with the hidden state $\mathbf{x}_l \in \mathbb{R}^{n \times C}$, which represents the $n$ parallel residual streams.

1.  **Flatten and Normalize:** The input matrix $\mathbf{x}_l$ is first flattened into a single vector $\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$ to capture the full context of all streams. This vector is then normalized using `RMSNorm`.
    \$
    \vec{\mathbf{x}}_{l}^{\prime} = \mathbf{RMSNorm}(\vec{\mathbf{x}}_{l})
    \$
    where `RMSNorm` (Root Mean Square Layer Normalization) is a common normalization technique that stabilizes training.

2.  **Generate Dynamic and Static Mappings:** The unconstrained mappings, denoted with a tilde ($\tilde{\mathcal{H}}$), are computed by combining a dynamic (input-dependent) part and a static (learnable bias) part. The dynamic part is generated by a linear projection of the normalized input vector $\vec{\mathbf{x}}_{l}^{\prime}$.

    This is described by the following set of equations:
\$
\left\{ \begin{array} { l l } { \vec { \mathbf { x } } _ { l } ^ { \prime } = \mathbf { R M S N o r m } ( \vec { \mathbf { x } } _ { l } ) } \\ { \mathcal { \tilde { H } } _ { l } ^ { \mathrm { p r e } } = \alpha _ { l } ^ { \mathrm { p r e } } \cdot ( \vec { \mathbf { x } } _ { l } ^ { \prime } \boldsymbol { \varphi } _ { l } ^ { \mathrm { p r e } } ) + \mathbf { b } _ { l } ^ { \mathrm { p r e } } } \\ { \mathcal { \tilde { H } } _ { l } ^ { \mathrm { p o s t } } = \alpha _ { l } ^ { \mathrm { p o s t } } \cdot ( \vec { \mathbf { x } } _ { l } ^ { \prime } \boldsymbol { \varphi } _ { l } ^ { \mathrm{post} } ) + \mathbf { b } _ { l } ^ { \mathrm { p o s t } } } \\ { \mathcal { \tilde { H } } _ { l } ^ { \mathrm { r e s } } = \alpha _ { l } ^ { \mathrm { r e s } } \cdot \mathrm { m a t } ( \vec { \mathbf { x } } _ { l } ^ { \prime } \boldsymbol { \varphi } _ { l } ^ { \mathrm { r e s } } ) + \mathbf { b } _ { l } ^ { \mathrm { r e s } } , } \end{array} \right.
\$
where:
*   $\vec{\mathbf{x}}_{l}^{\prime}$ is the normalized and flattened input.
*   $\alpha_l^{\mathrm{pre}}$, $\alpha_l^{\mathrm{post}}$, $\alpha_l^{\mathrm{res}}$ are learnable scalar gating factors, initialized to small values to ensure the network starts close to a standard residual connection.
*   $\boldsymbol{\varphi}_l^{\mathrm{pre}} \in \mathbb{R}^{nC \times n}$, $\boldsymbol{\varphi}_l^{\mathrm{post}} \in \mathbb{R}^{nC \times n}$, and $\boldsymbol{\varphi}_l^{\mathrm{res}} \in \mathbb{R}^{nC \times n^2}$ are learnable projection matrices that generate the dynamic part of the mappings.
*   $\mathbf{b}_l^{\mathrm{pre}} \in \mathbb{R}^{1 \times n}$, $\mathbf{b}_l^{\mathrm{post}} \in \mathbb{R}^{1 \times n}$, and $\mathbf{b}_l^{\mathrm{res}} \in \mathbb{R}^{n \times n}$ are learnable static bias terms.
*   $\mathrm{mat}(\cdot)$ is a function that reshapes the output vector of size $1 \times n^2$ into a matrix of size $n \times n$ for $\tilde{\mathcal{H}}_l^{\mathrm{res}}$.

    At this point, we have the unconstrained mappings $\tilde{\mathcal{H}}_l^{\mathrm{pre}}$, $\tilde{\mathcal{H}}_l^{\mathrm{post}}$, and $\tilde{\mathcal{H}}_l^{\mathrm{res}}$.

### 4.2.2. Manifold Projection
The next step is to project these unconstrained mappings onto the desired manifolds to ensure stability and non-negativity.

The final constrained mappings are obtained via:
\$
\left\{ \begin{array} { l l } { \mathcal { H } _ { l } ^ { \mathrm { p r e } } = \sigma ( \tilde { \mathcal { H } } _ { l } ^ { \mathrm { p r e } } ) } \\ { \mathcal { H } _ { l } ^ { \mathrm { p o s t } } = 2 \sigma ( \tilde { \mathcal { H } } _ { l } ^ { \mathrm { p o s t } } ) } \\ { \mathcal { H } _ { l } ^ { \mathrm { r e s } } = \mathrm { S i n k h o r n – K n o p p } ( \tilde { \mathcal { H } } _ { l } ^ { \mathrm { r e s } } ) , } \end{array} \right.
\$
where:
*   $\sigma(\cdot)$ is the **Sigmoid function**, which maps any real number to the range (0, 1). This is used to enforce non-negativity on the aggregation ($\mathcal{H}_l^{\mathrm{pre}}$) and distribution ($\mathcal{H}_l^{\mathrm{post}}$) vectors. The factor of 2 for $\mathcal{H}_l^{\mathrm{post}}$ scales the output range to (0, 2).
*   $\mathrm{Sinkhorn–Knopp}(\cdot)$ is the operator that projects the residual matrix $\tilde{\mathcal{H}}_l^{\mathrm{res}}$ onto the manifold of doubly stochastic matrices.

### 4.2.3. The Sinkhorn-Knopp Algorithm
This algorithm is an iterative process that converts a square matrix with positive entries into a doubly stochastic matrix.

1.  **Exponentiation:** First, the input matrix $\tilde{\mathcal{H}}_l^{\mathrm{res}}$ is made strictly positive by element-wise exponentiation.
    \$
    \mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_l^{\mathrm{res}})
    \$

2.  **Iterative Normalization:** The algorithm then repeatedly alternates between normalizing the rows to sum to 1 and normalizing the columns to sum to 1. The update rule for one iteration is:
    \$
    \mathbf{M}^{(t)} = \mathcal{T}_{r}(\mathcal{T}_{c}(\mathbf{M}^{(t-1)}))
    \$
    where:
    *   $\mathcal{T}_{c}(\mathbf{A})$ denotes column normalization: each element is divided by its column's sum.
    *   $\mathcal{T}_{r}(\mathbf{A})$ denotes row normalization: each element is divided by its row's sum.

        This process is proven to converge to a unique doubly stochastic matrix. In practice, the authors run it for a fixed number of iterations, $t_{\mathrm{max}} = 20$, which provides a close approximation. The final output is the constrained residual matrix $\mathcal{H}_l^{\mathrm{res}}$.

### 4.2.4. Efficient Infrastructure Design
A key part of the paper is making mHC practical. The authors detail three system-level optimizations:

1.  **Kernel Fusion:** To combat the memory I/O bottleneck, multiple consecutive operations are "fused" into a single GPU kernel. This avoids writing intermediate results to and then reading them back from GPU memory. They use `TileLang`, a programming model for AI systems, to implement these fused kernels. For example, the calculation of the unconstrained mappings and the subsequent projections are fused. They also fuse the residual merge step to reduce memory access from $(3n+1)C$ reads to $(n+1)C$ reads.

2.  **Recomputing:** The expanded $n$-stream residual state significantly increases the memory required to store activations for the backward pass. To mitigate this, mHC uses a recomputation strategy (a form of gradient checkpointing). The intermediate activations of the mHC kernels are discarded after the forward pass and recomputed on-the-fly during the backward pass. The paper provides a formula to find the optimal number of layers ($L_r$) to group into a recomputation block to minimize total memory usage:
    \$
    L_{r}^{*} = \arg\min_{L_r} \left[ nC \times \left\lceil \frac{L}{L_r} \right\rceil + (n+2)C \times L_r \right] \approx \sqrt{\frac{nL}{n+2}}
    \$
    where $L$ is the total number of layers.

3.  **Overlapping Communication in DualPipe:** In large-scale training with pipeline parallelism, the expanded residual stream increases communication between GPUs. The authors extend the `DualPipe` scheduling scheme to better overlap this communication with computation. As shown in Figure 4, they use dedicated high-priority compute streams and avoid long-running persistent kernels to ensure that communication can proceed without being blocked by computation, maximizing hardware utilization.

    The overall model structure with mHC is illustrated in Figure 1(c).

    ![Figure 1 | Illustrations of Residual Connection Paradigms. This figure compares the structural design of (a) standard Residual Connection, (b) Hyper-Connections (HC), and (c) our proposed Manifold-Constrained Hyper-Connections $( m \\mathbf { H } \\mathbf { C } )$ . Unlike the unconstrained HC, mHC focuses on optimizing the residual connection space by projecting the matrices onto a constrained manifold to ensure stability.](images/1.jpg)
    *该图像是示意图，展示了三种残差连接的结构设计，分别为 (a) 标准残差连接、(b) 超连接 (HC) 和 (c) 我们提出的流形约束超连接 `m extbf{HC}`。mHC通过将残差连接空间投影到受限流形上，从而优化了结构设计，确保了稳定性。*

# 5. Experimental Setup

## 5.1. Datasets
The paper evaluates the proposed methods on large-scale language model pre-training. However, the exact composition of the training data is not specified. The authors state:
*   For the main 3B, 9B, and 27B models, they use a "dataset size proportional to its parameters," following compute-optimal scaling principles.
*   For the token scaling experiment, they use a "fixed corpus of 1 trillion tokens."

    While the lack of specificity about the datasets (e.g., Common Crawl, C4, etc.) is a minor weakness for full reproducibility, the scales involved (up to 262B tokens for the 27B model and 1T tokens for the scaling experiment) are representative of modern LLM pre-training.

## 5.2. Evaluation Metrics
The models are evaluated on a range of downstream benchmarks. The metrics used are `Exact Match (EM)`, `F1 Score`, and `Accuracy (Acc)`.

### 5.2.1. Exact Match (EM)
*   **Conceptual Definition:** This metric measures the percentage of predictions that are an exact string match to one of the ground-truth answers. It is a strict metric commonly used in question answering and math problem-solving tasks where the answer must be precise.
*   **Mathematical Formula:**
    \$
    \text{EM} = \frac{\text{Number of predictions with exact match}}{\text{Total number of questions}}
    \$
*   **Symbol Explanation:** A prediction is an "exact match" if it is identical to a reference answer after minor normalization (e.g., lowercasing, removing punctuation).

### 5.2.2. F1 Score
*   **Conceptual Definition:** Used for tasks like extractive question answering (e.g., DROP), where the answer is a span of text from a given context. The F1 score is the harmonic mean of precision and recall, providing a balanced measure of a model's ability to identify all the correct words in an answer (recall) without including incorrect ones (precision).
*   **Mathematical Formula:**
    \$
    F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
    \$
*   **Symbol Explanation:**
    *   **Precision:** The proportion of words in the predicted answer that are also in the ground-truth answer.
        \$
        \text{Precision} = \frac{|\text{Predicted} \cap \text{Ground Truth}|}{|\text{Predicted}|}
        \$
    *   **Recall:** The proportion of words in the ground-truth answer that are also in the predicted answer.
        \$
        \text{Recall} = \frac{|\text{Predicted} \cap \text{Ground Truth}|}{|\text{Ground Truth}|}
        \$

### 5.2.3. Accuracy (Acc.)
*   **Conceptual Definition:** This is the standard classification accuracy, measuring the fraction of correct predictions out of the total number of samples. It's used for multiple-choice tasks like MMLU and HellaSwag.
*   **Mathematical Formula:**
    \$
    \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
    \$
*   **Symbol Explanation:** A "correct prediction" is one where the model's chosen option matches the correct label.

## 5.3. Baselines
The experiments compare three model variants:
1.  **Baseline:** A strong Mixture-of-Experts (MoE) model based on the `DeepSeek-V3` architecture. This model uses a standard single-stream residual connection.
2.  **HC (Hyper-Connections):** The baseline model augmented with the original, unconstrained `Hyper-Connections` architecture.
3.  **mHC (Manifold-Constrained Hyper-Connections):** The baseline model augmented with the proposed `mHC` architecture.

    For both HC and mHC, the residual stream expansion rate is set to $n=4$. The detailed model configurations for the 3B, 9B, and 27B variants are provided in Table 5 in the appendix.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Training Stability
The paper's primary claim is that mHC solves the instability of HC. Figure 5 provides strong evidence for this.

![Figure 5 | Training Stability of Manifold-Constrained Hyper-Connections $\\textstyle ( m \\mathbf { H } \\mathbf { C } )$ . This figure illustrates (a) the absolute training loss gap of mHC and HC relative to the baseline, and (b) the gradient norm of the three methods. All experiments utilize the 27B model. The results demonstrate that mHC exhibits improved stability in terms of both loss and gradient norm.](images/5.jpg)
*该图像是图表，展示了 mHC 和 HC 相较于基线的（a）绝对训练损失差距及（b）梯度范数随训练步骤变化的情况。实验结果表明，mHC 在损失和梯度范数上均展现了更好的稳定性。*

*   **Loss Curve (Figure 5a):** The plot shows the absolute training loss gap relative to the baseline. While both HC and mHC achieve a lower loss than the baseline, the curve for HC (orange) shows a sudden, large spike around the 12k training step. This indicates a moment of severe training instability. In contrast, the curve for mHC (blue) is smooth and consistently shows improvement, demonstrating stable training dynamics. The final loss reduction for mHC is 0.021 compared to the baseline.
*   **Gradient Norm (Figure 5b):** This plot directly measures the magnitude of gradients during training. The gradient norm for the baseline (green) and mHC (blue) are stable and closely aligned. However, the gradient norm for HC (orange) exhibits extreme volatility, correlating with the loss spike. This confirms that HC suffers from exploding gradients, a problem that mHC successfully mitigates.

### 6.1.2. Downstream Benchmark Performance
Table 4 presents the performance of the 27B models on various downstream tasks. It shows that the stability of mHC does not come at the cost of performance; in fact, it often improves it.

The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>Benchmark (Metric)</th>
<th>BBH (EM)</th>
<th>DROP (F1)</th>
<th>GSM8K (EM)</th>
<th>HellaSwag (Acc.)</th>
<th>MATH (EM)</th>
<th>MMLU (Acc.)</th>
<th>PIQA (Acc.)</th>
<th>TriviaQA (EM)</th>
</tr>
<tr>
<th># Shots</th>
<th>3-shot</th>
<th>3-shot</th>
<th>8-shot</th>
<th>10-shot</th>
<th>4-shot</th>
<th>5-shot</th>
<th>0-shot</th>
<th>5-shot</th>
</tr>
</thead>
<tbody>
<tr>
<td>27B Baseline</td>
<td>43.8</td>
<td>47.0</td>
<td>46.7</td>
<td>73.7</td>
<td>22.0</td>
<td>59.0</td>
<td>78.5</td>
<td>54.3</td>
</tr>
<tr>
<td>27B w/ HC</td>
<td>48.9</td>
<td>51.6</td>
<td>53.2</td>
<td>74.3</td>
<td>26.4</td>
<td>63.0</td>
<td>79.9</td>
<td>56.3</td>
</tr>
<tr>
<td>27B w/ mHC</td>
<td><strong>51.0</strong></td>
<td><strong>53.9</strong></td>
<td><strong>53.8</strong></td>
<td><strong>74.7</strong></td>
<td>26.0</td>
<td><strong>63.4</strong></td>
<td><strong>80.5</strong></td>
<td><strong>57.6</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **mHC vs. Baseline:** mHC consistently and significantly outperforms the baseline across all 8 benchmarks.
*   **mHC vs. HC:** mHC surpasses HC on 6 out of 8 benchmarks. Notably, the improvements are largest on complex reasoning tasks like `BBH` (+2.1% EM) and `DROP` (+2.3% F1). This suggests that the stabilized information flow in mHC may foster better development of reasoning capabilities. The only task where HC slightly outperforms mHC is `MATH`, though the difference is small.

### 6.1.3. Scaling Properties
Figure 6 demonstrates that the benefits of mHC are not limited to a specific scale but hold up as models and datasets grow.

![Figure 6 | Scaling properties of mHC compared to the Baseline. (a) Compute Scaling Curve. Solid lines depict the performance gap across different compute budgets. Each point represents a specific compute-optimal configuration of model size and dataset size, scaling from 3B and 9B to 27B parameters. (b) Token Scaling Curve. Trajectory of the 3B model during training. Each point represents the model's performance at different training tokens. Detailed architectures and training configurations are provided in Appendix A.1.](images/6.jpg)
*该图像是图表，展示了mHC与基线模型在计算和训练令牌的缩放特性。左侧的(a)计算缩放曲线显示了不同计算预算下的绝对损失差距，右侧的(b)令牌缩放曲线则表现了模型在不同训练令牌下的损失比率。*

*   **Compute Scaling (Figure 6a):** This plot shows the relative loss improvement of mHC over the baseline for 3B, 9B, and 27B models. The performance advantage is robust and only slightly decreases at the 27B scale, indicating that mHC scales well with model size and compute budget.
*   **Token Scaling (Figure 6b):** This plot tracks the performance of a 3B model trained on 1 trillion tokens. The loss ratio of mHC to the baseline remains consistently favorable throughout training, confirming that the benefits are sustained over a long training horizon and are not just an early-stage artifact.

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Importance of Residual Mapping
The paper first ablates the components of the original HC architecture to justify its focus on the $H_l^res$ matrix.

The following are the results from Table 1 of the original paper:

| Hres | Hpr | Hpost | Absolute Loss Gap |
| :--: | :-: | :---: | :---------------: |
|      |     |       | 0.0               |
|  ✓   |     |       | -0.022            |
|  ✓   |  ✓  |       | -0.025            |
|  ✓   |  ✓  |   L   | -0.027            |

*Note: The paper seems to have a typo in the last row, using 'L' instead of '✓'. Assuming 'L' means '✓'.*
**Analysis:** This table shows that adding the learnable residual mapping `Hres` alone accounts for the vast majority of the performance gain (-0.022 loss gap). Adding `Hpre` and `Hpost` provides only marginal additional benefits. This ablation study strongly justifies the paper's focus on analyzing and fixing the `Hres` component.

### 6.2.2. Propagation Stability Analysis
This is the most direct verification of the mHC mechanism. The authors analyze the "Amax Gain Magnitude," which measures the maximum amplification factor for signals (forward pass) and gradients (backward pass).

**Unstable Propagation in HC (Figure 3):**

![Figure 3 | Propagation Instability of Hyper-Connections (HC). This figure illustrates the propagation dynamics of (a) the single-layer mapping $\\mathcal { H } _ { l } ^ { \\mathrm { r e s } }$ ad (b) the composite mapping $\\Pi _ { i = 1 } ^ { L - l } \\mathcal { H } _ { L - i } ^ { \\mathrm { r e s } }$ $\\mathbf { \\hat { x } } \\mathbf { \\cdot }$ block into two independent layers (Attention and FFN). The Amax Gain Magnitude (y-axis) is calculated as the maximum absolute row sum (for the forward signal) and column sum (for the backward gradient), averaged over all tokens in a selected sequence.](images/3.jpg)
*该图像是图表，展示了超连接（HC）传播的不稳定性。从左侧的图(a)中可以看到单层映射 $\mathcal{H}_{l}^{\text{res}}$ 的前向信号增益与反向梯度增益的变化；而右侧的图(b)展示了复合映射 $\Pi_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$ 的前向信号增益和反向梯度增益。y轴表示Amax增益幅度。*

*   For HC, the composite mapping gain (Figure 3b) shows extreme values, peaking at **3000**. This is a clear visualization of exploding signals/gradients, confirming the theoretical problem of multiplying unconstrained matrices.

**Stable Propagation in mHC (Figure 7):**

![Figure 7 | Propagation Stability of Manifold-Constrained Hyper-Connections $\\textstyle ( m \\mathbf { H } \\mathbf { C } )$ . This figure illustrates the propagation dynamics of (a) the single-layer mapping $\\mathcal { P } _ { M ^ { \\mathrm { r e s } } } ( \\mathcal { H } _ { l } ^ { \\mathrm { r e s } } )$ and (b) the composite mapping $\\Pi _ { i = 1 } ^ { L - l } { \\mathcal { P } } _ { M ^ { \\mathrm { r e s } } } ( { \\mathcal { H } } _ { L - i } ^ { \\mathrm { r e s } } )$ w h me Te sult demott mHC significantly enhances propagation stability compared to HC.](images/7.jpg)
*该图像是一个图表，展示了流形约束超连接（mHC）的传播稳定性。左侧（a）为单层映射 $\mathcal{P}_{M^{\mathrm{res}}}(\mathcal{H}_{l}^{\mathrm{res}})$ 的信号增益，而右侧（b）为复合映射 $\Pi_{i=1}^{L-l} \mathcal{P}_{M^{\mathrm{res}}}(\mathcal{H}_{L-i}^{\mathrm{res}})$ 的信号增益。图中显示，mHC 在前向信号增益和反向梯度增益方面均显著提升了传播稳定性。*

*   For mHC, the single-layer gain (Figure 7a) is tightly centered around 1, as expected from the doubly stochastic constraint.
*   Crucially, the composite mapping gain (Figure 7b) remains bounded, reaching a maximum of only **~1.6**. This is a dramatic improvement of three orders of magnitude compared to HC's 3000. It provides definitive proof that the manifold constraint successfully tames the signal propagation and ensures stability.

**Visualization of Learned Mappings (Figure 8):**

![Figure 8 | Visualizations of Learnable Mappings. This figure displays representative singlelayer and composite mappings for HC (first row) and mHC (second row). Each matrix is computed by averaging over all tokens within a selected sequence. The labels annotated along the y-axis and x-axis indicate the forward signal gain (row sum) and the backward gradient gain (column sum), respectively.](images/8.jpg)
*该图像是一个图表，展示了 HC 和 mHC 的代表性单层和复合映射。矩阵中的数值是通过对选定序列中所有标记的平均值得到的，y 轴和 x 轴的标签分别表示前向信号增益（行和）和后向梯度增益（列和）。*

This figure visualizes the learned matrices themselves.
*   **HC (top row):** The matrices for HC show large and erratic values, with some row/column sums (gains) being far from 1. This reflects their unconstrained and unstable nature.
*   **mHC (bottom row):** The matrices for mHC are visually very different. They are well-structured, with all values being non-negative and the row/column sums very close to 1. This confirms that the Sinkhorn-Knopp algorithm is effectively enforcing the doubly stochastic constraint during training.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully identifies a critical flaw in the promising `Hyper-Connections (HC)` architecture: its unconstrained nature compromises the `identity mapping` property, leading to severe training instability and hampering scalability.

The authors propose an elegant and effective solution, **Manifold-Constrained Hyper-Connections (mHC)**. By projecting the residual mixing matrix onto the manifold of doubly stochastic matrices, mHC restores a generalized, stable form of identity mapping. This transforms the residual update from an arbitrary linear transformation into a well-behaved convex combination of features.

Empirical results on large-scale language models demonstrate that mHC not only resolves the instability of HC but also leads to superior performance, particularly on reasoning tasks. Furthermore, by co-designing the architecture with rigorous infrastructure optimizations (`kernel fusion`, `recomputing`), the authors show that these benefits can be achieved with a minimal `6.7%` computational overhead, making mHC a practical and scalable solution for training next-generation foundation models.

## 7.2. Limitations & Future Work
The authors themselves point out several promising directions for future research:
*   **Exploration of Other Manifolds:** The paper uses doubly stochastic matrices, but the mHC framework is general. Future work could explore other types of matrix manifolds that might offer different trade-offs between expressive power (plasticity) and training stability, tailored to specific tasks or objectives.
*   **Revitalizing Macro-Architecture Research:** The authors hope that this work will encourage more research into the topological design of neural networks, moving beyond the standard micro-design of attention and FFN blocks.

    From a critical standpoint, some minor limitations include:
*   **Lack of Dataset Specificity:** The pre-training datasets are described in general terms of size but not by name, which slightly hinders exact reproducibility.
*   **Generalization to Other Architectures:** The experiments are conducted on `DeepSeek-V3`-style MoE models. While this is a strong and relevant testbed, demonstrating the effectiveness of mHC on other popular LLM families (like Llama or GPT) would further strengthen its claims of generality.

## 7.3. Personal Insights & Critique
This paper is an excellent example of rigorous, problem-driven research that combines deep theoretical insight with practical engineering.

**Key Strengths:**
1.  **Clear Problem Diagnosis:** The paper does not just propose a new method but starts by clearly diagnosing a fundamental problem (instability from broken identity mapping) in a recent, high-performing architecture. The analysis in Figure 3 is compelling.
2.  **Theoretically-Grounded Solution:** The choice of the doubly stochastic manifold is not arbitrary; it is motivated by deep mathematical properties (norm preservation, compositional closure) that directly address the diagnosed problem. This makes the solution elegant and principled.
3.  **Holistic, Full-Stack Approach:** A standout feature of this paper is its attention to system-level efficiency. Many academic papers on novel architectures ignore the practical costs of memory access and computation. By including detailed infrastructure optimizations, the authors make their proposal immediately viable for real-world, large-scale applications. This is a model for how architectural research should be done in the era of foundation models.
4.  **Strong Empirical Evidence:** The experiments are comprehensive, covering stability, final performance, and scaling properties. The direct comparison between Figure 3 (HC) and Figure 7 (mHC) is a "slam dunk" piece of evidence for the method's effectiveness.

**Potential Reflections:**
*   The work beautifully illustrates the tension between **expressivity** and **stability** in deep learning. Unconstrained models (like HC) are more expressive but risk instability. The original ResNet chose maximum stability. mHC finds a "sweet spot" in between, allowing for learnable, expressive interactions while maintaining guaranteed stability.
*   This paper could be seen as part of a broader trend of re-examining the foundational principles of deep learning architectures in the context of massive scale. As models become deeper and larger, subtle instabilities that were manageable at smaller scales can become catastrophic. Principled constraints, like those in mHC, are becoming increasingly important.
*   The success of mHC suggests that the future of architecture design may lie in finding the right **inductive biases** to impose on network topology and transformations, rather than relying on purely unconstrained learning. The choice of manifold is a powerful way to inject such biases.