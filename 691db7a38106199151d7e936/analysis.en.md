# 1. Bibliographic Information

## 1.1. Title
Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation

## 1.2. Authors
Hongyu Lu (WeChat, Tencent), Bowen Zheng (Renmin University of China), Junjie Zhang (Renmin University of China), Yu Chen (WeChat, Tencent), Ming Chen (WeChat, Tencent), Wayne Xin Zhao (Renmin University of China), Ji-Rong Wen (Renmin University of China).
The authors are affiliated with prominent technology companies (Tencent) and academic institutions (Renmin University of China), indicating a strong research background in areas like recommendation systems, graph neural networks, and contrastive learning.

## 1.3. Journal/Conference
The paper is published in the proceedings of ACM (Conference acronym 'XX'). Given the context and authors' previous publications, it is likely a top-tier conference in information retrieval or data mining, such as SIGIR or KDD, known for their significant influence in the field of recommender systems.

## 1.4. Publication Year
2024

## 1.5. Abstract
Graph neural network (GNN) has been a powerful approach in collaborative filtering (CF) due to its ability to model high-order user-item relationships. Recently, to alleviate the data sparsity and enhance representation learning, many efforts have been conducted to integrate contrastive learning (CL) with GNNs. Despite the promising improvements, the contrastive view generation based on structure and representation perturbations in existing methods potentially disrupts the collaborative information in contrastive views, resulting in limited effectiveness of positive alignment. To overcome this issue, we propose CoGCL, a novel framework that aims to enhance graph contrastive learning by constructing contrastive views with stronger collaborative information via discrete codes. The core idea is to map users and items into discrete codes rich in collaborative information for reliable and informative contrastive view generation. To this end, we initially introduce a multi-level vector quantizer in an end-to-end manner to quantize user and item representations into discrete codes. Based on these discrete codes, we enhance the collaborative information of contrastive views by considering neighborhood structure and semantic relevance respectively. For neighborhood structure, we propose virtual neighbor augmentation by treating discrete codes as virtual neighbors, which expands an observed user-item interaction into multiple edges involving discrete codes. Regarding semantic relevance, we identify similar users/items based on shared discrete codes and interaction targets to generate the semantically relevant view. Through these strategies, we construct contrastive views with stronger collaborative information and develop a triple-view graph contrastive learning approach. Extensive experiments on four public datasets demonstrate the effectiveness of our proposed approach.

## 1.6. Original Source Link
https://arxiv.org/abs/2409.05633
PDF Link: https://arxiv.org/pdf/2409.05633v2.pdf
Publication Status: Preprint on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The paper addresses the critical challenge of enhancing recommender systems, particularly `collaborative filtering (CF)`, by leveraging `Graph Neural Networks (GNNs)`. GNNs are effective at modeling complex, high-order user-item relationships, but they often struggle with `data sparsity`—a common problem where user-item interaction data is scarce. This sparsity limits the quality of learned user and item representations, which are crucial for accurate recommendations.

To mitigate data sparsity and improve representation learning, `contrastive learning (CL)` has been integrated with GNNs. However, existing CL-based methods typically generate contrastive views through perturbations (e.g., `stochastic node/edge dropout` or `adding random noise` to embeddings). The core problem identified by the authors is that these perturbation-based approaches can unintentionally **disrupt the underlying collaborative information** within the generated contrastive views. This disruption leads to `ineffective positive alignment`, where the model is taught to align views that no longer fully represent the true collaborative semantics, thereby limiting the overall effectiveness of CL.

The paper's entry point is to overcome this limitation by proposing a novel method for contrastive view generation that **preserves and enhances collaborative information** rather than disrupting it. The innovative idea is to use `discrete codes` derived from user and item representations to construct more reliable and informative contrastive views.

## 2.2. Main Contributions / Findings
The paper introduces `CoGCL` (Collaborative Graph Contrastive Learning), a novel framework designed to enhance graph contrastive learning for recommendation by creating contrastive views with stronger collaborative information.

The primary contributions are:
1.  **A Reliable and Informative Graph CL Framework (CoGCL):** CoGCL is proposed as a method to construct contrastive views that inherently contain stronger collaborative information, moving beyond perturbation-based view generation.
2.  **End-to-End Discrete Code Learning:** An end-to-end multi-level `vector quantizer` is introduced to map continuous user and item representations (learned by a GNN) into discrete codes. These codes are specifically designed to capture and represent rich collaborative semantics.
3.  **Enhanced Contrastive View Generation via Discrete Codes:** The learned discrete codes are utilized in two innovative ways to create high-quality contrastive views:
    *   **Virtual Neighbor Augmentation:** Discrete codes are treated as "virtual neighbors," expanding existing user-item interactions into multiple edges involving these codes. This strategy enriches the `neighborhood structure` of nodes and `alleviates data sparsity` by providing more connections. This augmentation can either `replace` existing neighbors with codes or `add` codes as extra neighbors.
    *   **Semantic Relevance Sampling:** Users or items are identified as semantically similar if they share discrete codes or common interaction targets. This allows for positive pair sampling that focuses on `fine-grained semantic relevance` rather than arbitrary pairings.
4.  **Triple-View Graph Contrastive Learning:** The framework integrates three distinct contrastive views: two augmented views derived from virtual neighbor augmentation and one semantically relevant view from similar users/items. A `triple-view graph contrastive learning` approach is developed to align these views, thereby integrating both enhanced structural and semantic collaborative information into the model.

    The key findings demonstrate that CoGCL consistently outperforms state-of-the-art baseline models across four public datasets, achieving significant improvements in recommendation performance. Detailed analyses further confirm that the proposed components (virtual neighbor augmentation, semantic relevance sampling, and the discrete code learning) are crucial for enhancing graph CL, particularly in scenarios with high data sparsity. The study also empirically validates that CoGCL achieves a better balance between `alignment` and `uniformity` in representation learning compared to previous methods.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### Collaborative Filtering (CF)
`Collaborative Filtering (CF)` is a fundamental technique in recommender systems that predicts a user's interest in items by collecting preferences or taste information from many users (collaborators). The basic idea is that if two users have similar preferences for some items, they are likely to have similar preferences for other items. There are two main types:
*   **User-based CF:** Recommends items that similar users have liked.
*   **Item-based CF:** Recommends items that are similar to items a user has liked in the past.
    CF relies on the `user-item interaction matrix`, which records implicit (e.g., clicks, purchases) or explicit (e.g., ratings) feedback. The goal is to fill in the missing entries in this matrix to predict preferences for unseen items.

### Graph Neural Networks (GNNs)
`Graph Neural Networks (GNNs)` are a class of neural networks designed to operate on graph-structured data. Unlike traditional neural networks that work on Euclidean data (like images or text sequences), GNNs can process data where relationships between entities are explicitly represented as a graph. In recommender systems, `user-item interaction data` can be naturally modeled as a `bipartite graph` where users and items are nodes, and interactions are edges.
*   **Node Representation Learning:** GNNs learn `node embeddings` (vector representations) by iteratively aggregating information from a node's neighbors. This process is called `message passing` or `neighbor aggregation`.
*   **High-Order Relationships:** By stacking multiple GNN layers, information can propagate across several hops, effectively capturing `high-order relationships` (e.g., a user's preference for an item might be influenced by items liked by users who liked similar items).
*   **LightGCN:** A simplified GNN architecture commonly used in recommendation. It removes non-linear activation functions and feature transformations from traditional GCNs, arguing that these components can actually hurt performance in collaborative filtering tasks due to over-smoothing or introducing noise. `LightGCN` focuses purely on neighbor aggregation, essentially performing a weighted sum of neighbor embeddings. The embedding of a node at layer $l$ is an aggregation of its neighbors' embeddings from layer `l-1`. The final node embedding is typically a concatenation or summation of embeddings from all layers.

### Contrastive Learning (CL)
`Contrastive Learning (CL)` is a self-supervised learning paradigm where the model learns representations by distinguishing between similar (positive) and dissimilar (negative) pairs of data samples. The core idea is to pull `positive pairs` (different augmented views of the same data point, or semantically similar data points) closer in the embedding space while pushing `negative pairs` (dissimilar data points) farther apart.
*   **Contrastive Views:** These are different augmented versions of an original data sample. How these views are generated is crucial for CL's effectiveness.
*   **InfoNCE Loss:** A common objective function used in CL. For a given anchor $z$, a positive sample $z+$, and a set of negative samples `z-`, the `InfoNCE` loss aims to maximize the similarity between $z$ and $z+$ relative to the similarity between $z$ and any `z-`.
    The `InfoNCE` loss for a sample $v$ with two views $\mathbf{z}_v'$ and $\mathbf{z}_v''$ is defined as:
    $$
    \mathcal{L}_{cl} = - \log \frac { e ^ { s ( \mathbf { z } _ { v } ^ { \prime } , \mathbf { z } _ { v } ^ { \prime \prime } ) / \tau } } { e ^ { s ( \mathbf { z } _ { v } ^ { \prime } , \mathbf { z } _ { v } ^ { \prime \prime } ) / \tau } + \sum _ { \tilde { v } \in \mathcal { V } _ { \mathrm { n e g } } } e ^ { s ( \mathbf { z } _ { v } ^ { \prime } , \mathbf { z } _ { \tilde { v } } ^ { \prime \prime } ) / \tau } }
    $$
    Where:
    *   $s(\cdot, \cdot)$: A similarity function, typically `cosine similarity`.
    *   $\tau$: A `temperature coefficient` that scales the logits before the softmax operation, controlling the sharpness of the distribution. A smaller $\tau$ makes the model more sensitive to small differences in similarity.
    *   $\mathbf{z}_v'$ and $\mathbf{z}_v''$: The representations of the two contrastive views for node $v$.
    *   $\mathcal{V}_{\mathrm{neg}}$: A set of `negative samples` for node $v$.
*   **Alignment and Uniformity:** These are two key properties of learned representations in CL, especially relevant for `InfoNCE` loss.
    *   `Alignment`: Measures how close positive pairs are in the embedding space. Good alignment means $s(\mathbf{z}, \mathbf{z}^+)$ is high.
    *   `Uniformity`: Measures how uniformly the representations are distributed on the unit hypersphere. Good uniformity means representations are spread out, preventing `representation collapse` (where all embeddings become similar). The `InfoNCE` loss implicitly encourages both.

### Vector Quantization (VQ)
`Vector Quantization (VQ)` is a signal processing technique used to reduce the data rate by mapping vectors from a large vector space to a finite number of regions in that space. Each region is represented by a `codevector` (or `codebook entry`), and the collection of all codevectors forms a `codebook`.
*   **Discrete Codes:** The output of VQ is a discrete index (or a sequence of indices) corresponding to the chosen codevector(s), effectively discretizing the continuous input space.
*   **Multi-level VQ:** Techniques like `Residual Quantization (RQ)` and `Product Quantization (PQ)` are multi-level VQ methods.
    *   **Residual Quantization (RQ):** Quantizes a vector in multiple stages. In each stage, a codevector is chosen to best represent the *residual* (the difference between the original vector and the sum of previously chosen codevectors). This allows for progressive refinement and potentially better accuracy with fewer codevectors per stage.
    *   **Product Quantization (PQ):** Divides the original vector into several sub-vectors and quantizes each sub-vector independently using its own small codebook. The final code is a concatenation of the codes for each sub-vector. This is efficient for high-dimensional vectors.

## 3.2. Previous Works
The paper frames its contribution by contrasting with existing approaches in two main areas: GNN-based CF and CL-based methods for recommendation.

### Traditional Collaborative Filtering (CF) and GNN-based CF
*   **BPR [36]:** A foundational `matrix factorization (MF)` approach that optimizes for a personalized ranking by sampling positive and negative item pairs.
*   **GCMC [41]:** `Graph Convolutional Matrix Completion` uses GCNs to perform matrix completion for recommendations.
*   **NGCF [48]:** `Neural Graph Collaborative Filtering` explicitly models high-order connectivity in the user-item interaction graph using GNNs.
*   **DGCF [49]:** `Disentangled Graph Collaborative Filtering` aims to learn disentangled representations to capture different user intents or item aspects, thereby improving recommendation quality.
*   **LightGCN [15]:** Simplifies GNNs for recommendation by removing non-linear activations and feature transformations, focusing purely on linear message passing, which has proven highly effective.
*   **SimpleX [31]:** A strong baseline that leverages a `cosine contrastive loss` for learning user/item embeddings without complex graph structures or CL augmentations.

### Contrastive Learning (CL) for Recommendation
Existing CL-based methods are categorized by how they construct contrastive views:
*   **Structure Augmentation:** These methods perturb the `graph structure` to create different views.
    *   **SGL [51]:** `Self-supervised Graph Learning` for recommendation. It randomly drops nodes or edges in the interaction graph to create augmented graphs, from which two views are derived. The paper critiques this as potentially disrupting crucial collaborative information, especially in sparse graphs.
    *   **GFormer [27]:** `Graph Transformer` for recommendation that distills self-supervised signals using graph rationale discovery based on masked autoencoding.
    *   **LightGCL [4]:** Employs `Singular Value Decomposition (SVD)` for adjacency matrix reconstruction to generate augmented views, aiming for a lightweight approach.
*   **Representation Augmentation:** These methods perturb or generate additional `node representations`.
    *   **SLRec [56]:** Uses CL for representation regularization to learn better latent relationships in general item recommendations.
    *   **NCL [29]:** `Neighborhood-enriched Contrastive Learning` enhances GNN-based recommendation by learning cluster centers based on an `Expectation-Maximization (EM)` algorithm as anchors.
    *   **HCCF [53]:** `Hypergraph Contrastive Collaborative Filtering` constructs hypergraph-enhanced CL to capture local and global collaborative relations.
    *   **SimGCL [60]:** Creates contrastive views by adding `random noise` to the node embeddings directly in the embedding space. This is a very simple yet effective method, but the paper argues that random noise can interfere with implicit collaborative semantics.

### User/Item ID Discretization in Recommendation
*   This area focuses on representing users/items not just by a single ID, but by a tuple of discrete codes.
*   **Semantic Hashing [5, 19, 37]** and **Vector Quantization [14, 44]** are key techniques used here.
*   Early work focused on efficiency (memory/time) by sharing code embeddings [1, 24, 25, 28, 38].
*   More recently, discrete codes are used to improve recommendation quality by alleviating sparsity and providing prior semantics, e.g., in transferable recommendation [16], generative sequential recommendation [30, 34, 39, 47], and LLM-based recommendation [18, 64].
*   The current paper's approach differs by specifically employing discrete codes for `virtual neighbor augmentation` and `semantic similarity sampling` within a graph CL framework for CF.

## 3.3. Technological Evolution
The evolution of recommender systems has seen a progression from traditional matrix factorization techniques to graph-based methods, and more recently, the integration of self-supervised learning paradigms like contrastive learning.
1.  **Early CF (e.g., BPR):** Focused on learning latent factors for users and items, primarily addressing the `cold-start problem` and `sparsity` in a basic sense.
2.  **GNN-based CF (e.g., NGCF, LightGCN):** Recognized the graph nature of user-item interactions and leveraged GNNs to explicitly model `high-order connectivity` and `information propagation`, leading to richer representations. This marked a significant leap in capturing complex relationships.
3.  **CL-enhanced GNN-based CF (e.g., SGL, SimGCL):** Introduced `self-supervised signals` via contrastive learning to further alleviate sparsity and enhance representation quality. These methods typically generate multiple "views" of the graph or node embeddings and maximize agreement between views of the same entity while pushing apart views of different entities.

    This paper's work (`CoGCL`) fits into the third stage. It identifies a crucial limitation in existing CL-enhanced methods: their view generation strategies (perturbations) can inadvertently degrade the very `collaborative information` they aim to leverage. CoGCL proposes a more sophisticated and `collaborative information-aware` view generation mechanism using discrete codes, positioning it as an advancement in making CL more effective and reliable for recommendation.

## 3.4. Differentiation Analysis
The core differentiation of CoGCL from existing CL-based methods lies in its approach to contrastive view generation:

| Feature                       | Existing CL Methods (e.g., SGL, SimGCL)                                     | CoGCL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| :---------------------------- | :---------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Contrastive View Generation** | - **Structure Perturbation:** Randomly drops nodes/edges (SGL), SVD-based reconstruction (LightGCL). - **Representation Perturbation:** Adds random noise to embeddings (SimGCL). | - **Discrete Codes:** Maps user/item representations to discrete codes rich in collaborative information. - **Virtual Neighbor Augmentation:** Uses discrete codes as virtual neighbors (`replace` or `add` existing neighbors) to expand interaction edges, enriching neighborhood structure and alleviating sparsity. - **Semantic Relevance Sampling:** Identifies similar users/items based on `shared discrete codes` or `shared interaction targets` to form positive pairs, emphasizing fine-grained semantic relevance. |
| **Impact on Collaborative Info** | - **Disruption Risk:** Perturbations can inadvertently disrupt or destroy crucial collaborative information, leading to less effective positive alignment and potentially misleading model learning. - **Arbitrary Nature:** Perturbations can be arbitrary and may not be well-founded in observed user-item interactions. | - **Preservation & Enhancement:** Explicitly aims to preserve and enhance collaborative information. Discrete codes are learned to be rich in this information. Augmentations are `reliable` (based on observed interactions) and `informative` (introducing richer structural and semantic context).                                                                                                                                                                                                                                                                                                                                 |
| **Positive Sample Definition** | - Views of the same node from perturbed graphs/embeddings. - May indiscriminately distinguish different instances. | - **Multi-faceted Positives:** Considers views from augmented graphs (abundant neighborhood structure) *and* semantically similar users/items (fine-grained semantic relevance) as positives. This leads to a `triple-view` approach. - Focuses on aligning instances with strong, explicit collaborative semantics, not just augmented versions of the same entity.                                                                                                                                                                                                                                                                                                                                   |
| **Robustness to Sparsity**    | - Structural perturbations on sparse graphs can lose key interactions. - Random noise might interfere with implicit semantics. | - Explicitly addresses sparsity through `virtual neighbor augmentation`, which adds reliable connections via discrete codes. This makes the augmented graphs richer even for sparse nodes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Learning Objective**        | - Primarily `InfoNCE` aiming for alignment and uniformity, often heavily relying on uniformity to prevent collapse. | - `Triple-view contrastive learning` that integrates `BPR loss`, `discrete code learning loss`, `alignment between neighbor augmented views`, and `alignment between semantically relevant users/items`. The paper empirically shows that alignment in CoGCL is more effective, not just relying on uniformity.                                                                                                                                                                                                                                                                                                                                                                                                                   |

In essence, while previous methods often apply generic data augmentation techniques, CoGCL proposes a **collaborative-information-centric** approach to augmentation by introducing discrete codes, thereby creating more meaningful and less disruptive contrastive views.

# 4. Methodology

## 4.1. Principles
The core principle of CoGCL is to overcome the limitations of existing `Graph Contrastive Learning (GCL)` methods for recommendation, which often disrupt collaborative information when generating contrastive views through perturbations. Instead, CoGCL aims to construct `reliable and informative` contrastive views that explicitly **enhance collaborative information** by leveraging `discrete codes`. The intuition is that if users and items can be represented by a small set of discrete codes that capture their underlying collaborative semantics, these codes can then be used to create more meaningful augmented interactions and identify semantically similar entities, thereby providing stronger self-supervision signals for representation learning.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Approach Overview
The CoGCL framework is built upon three main components:
1.  **End-To-End Discrete Code Learning:** This component focuses on how to elegantly learn discrete codes that are rich in collaborative information for both users and items. It uses a multi-level vector quantizer integrated directly into the training process.
2.  **Reliable and Informative Contrastive View Generation:** Once the discrete codes are learned, they are employed to create high-quality contrastive views. This involves two sub-strategies: `virtual neighbor augmentation` to enhance neighborhood structure and `semantic relevance sampling` to identify truly similar entities.
3.  **Triple-View Graph Contrastive Learning:** Finally, the framework develops a contrastive learning objective that aligns representations from three distinct, collaboratively-enhanced views, thereby integrating stronger self-supervised signals into the recommendation model.

### 4.2.2. End-To-End Discrete Code Learning

#### 4.2.2.1. Representation Encoding via GNN
Following established practices in GNN-based collaborative filtering, CoGCL adopts `LightGCN` as its primary `Graph Neural Network (GNN)` encoder. `LightGCN` is chosen for its simplicity and effectiveness in propagating neighbor information across the `user-item interaction graph`.

Given user and item sets $\mathcal{U}$ and $\mathcal{I}$ respectively, an interaction matrix $\mathbf{R} \in \{0, 1\}^{|\mathcal{U}| \times |\mathcal{I}|}$ defines interactions. A bipartite graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is constructed where $\mathcal{V} = \{\mathcal{U} \cup \mathcal{I}\}$ contains all users and items, and $\mathcal{E} = \{ (u, i) | u \in \mathcal{U}, i \in \mathcal{I}, \mathbf{R}_{u,i}=1 \}$ represents observed interactions.

The GNN encoder, specifically `LightGCN`, learns node representations by aggregating information from neighbors. The process is formulated as:
$$
\begin{array}{r}
\boldsymbol{\mathbf{Z}}^l = \mathrm{GNN}(\rho(\boldsymbol{\mathbf{Z}}^{l-1}), G), \quad \boldsymbol{\mathbf{Z}} \mathrm{~=~ \boldsymbol{\mathbf{Readout}}(\boldsymbol{\mathbf{Z}}^0, \boldsymbol{\mathbf{Z}}^1, \ldots, \boldsymbol{\mathbf{Z}}^L])
\end{array}
$$
Where:
*   $L$: The total number of `GNN layers`.
*   $\boldsymbol{\mathbf{Z}}^l \in \mathbb{R}^{|\mathcal{V}| \times d}$: The matrix of node representations at the $l$-th GNN layer, where $d$ is the embedding dimension. These representations capture $l$-hop neighbor information.
*   $\rho(\cdot)$: Denotes a `dropout operation` applied to the input representations of each layer. This is a regularization technique used to mitigate overfitting by randomly setting a fraction of input units to zero during training. Importantly, CoGCL applies dropout on the input representation of each layer, not as `edge dropout` on the graph structure, which is a common practice in some prior works.
*   $G$: Represents the `bipartite interaction graph`.
*   $\boldsymbol{\mathbf{Z}}^0$: The initial `trainable ID embedding matrix` for all users and items.
*   $\mathrm{Readout}(\cdot)$: A function that summarizes the representations from all layers ($\boldsymbol{\mathbf{Z}}^0, \boldsymbol{\mathbf{Z}}^1, \ldots, \boldsymbol{\mathbf{Z}}^L$) to obtain the final node representations $\boldsymbol{\mathbf{Z}}$. Following `SimGCL`, CoGCL specifically skips $\boldsymbol{\mathbf{Z}}^0$ in the readout function for slight performance improvement.
    The final user and item representations, denoted as $z_u$ and $z_i$, are then used for both the recommendation task and the multi-level discrete code learning. The predicted score for user $u$ and item $i$ is typically calculated as the `inner product` or `cosine similarity` of their representations, e.g., $\hat{y}_{ui} = z_u^T z_i$.

#### 4.2.2.2. End-To-End Multi-Level Code Learning
To obtain discrete codes that are rich in collaborative information, CoGCL employs a `multi-level vector quantization (VQ)` method in an end-to-end manner. The paper specifically mentions `Residual Quantization (RQ)` [9] and `Product Quantization (PQ)` [21] as examples of multi-level VQ. For simplicity, the process is described using a generic multi-level VQ framework, exemplified with RQ mechanics.

Let's consider discrete code learning for a user $u$. The process applies symmetrically to items.
For each level $h \in \{1, \ldots, H\}$, there is a `codebook` $C^h = \{\mathbf{e}_k^h\}_{k=1}^K$, which consists of $K$ distinct `codevectors` $\mathbf{e}_k^h$. Each $\mathbf{e}_k^h$ is a learnable embedding representing a specific discrete code at level $h$.

The quantization process, which assigns a discrete code $c_u^h$ to a user representation $\mathbf{z}_u^h$ at level $h$, is expressed using a `softmax` over `cosine similarity`:
$$
c_u^h = \underset{k}{\arg\operatorname*{max}} P(k | \mathbf{z}_u^h), \quad P(k | \mathbf{z}_u^h) = \frac { e ^ { s ( \mathbf { z } _ { u } ^ { h } , \mathbf { e } _ { k } ^ { h } ) / \tau } } { \sum _ { j = 1 } ^ { K } e ^ { s ( \mathbf { z } _ { u } ^ { h } , \mathbf { e } _ { j } ^ { h } ) / \tau } }
$$
Where:
*   $c_u^h$: The $h$-th discrete code assigned to user $u$.
*   $\mathbf{z}_u^h$: The user representation at the $h$-th level. In `Residual Quantization (RQ)`, $\mathbf{z}_u^1 = \mathbf{z}_u$ (the initial user representation from the GNN), and for subsequent levels, $\mathbf{z}_u^{h+1} = \mathbf{z}_u^h - \mathbf{e}_{c_h}^h$, meaning the residual from the previous quantization is passed to the next level. In `Product Quantization (PQ)`, $\mathbf{z}_u$ would be split into $H$ sub-vectors, and each $\mathbf{z}_u^h$ would be a sub-vector.
*   $\mathbf{e}_k^h$: The $k$-th codevector in the codebook $C^h$ at level $h$.
*   $s(\cdot, \cdot)$: The `cosine similarity` function, aligning with the similarity measure used in contrastive learning. This is a deliberate choice, differing from the `Euclidean distance` often used in traditional VQ, to synchronize the learning objectives.
*   $\tau$: A `temperature coefficient` for the softmax, similar to the one in `InfoNCE` loss.

    The learning objective for these discrete codes is to maximize the likelihood of assigning representations to their corresponding codebook centers. This is achieved via a `Cross-Entropy (CE)` loss. For user discrete code learning, the loss is:
$$
\mathcal { L } _ { c o d e } ^ { U } = - \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \log P ( c _ { u } ^ { h } | \mathbf { z } _ { u } ^ { h } )
$$
Where:
*   $\mathcal{L}_{code}^U$: The discrete code loss for the user side.
*   $H$: The number of code levels.
*   $P(c_u^h | \mathbf{z}_u^h)$: The probability that the user representation $\mathbf{z}_u^h$ is assigned to its chosen code $c_u^h$ at level $h$.
    A similar loss, $\mathcal{L}_{code}^I$, is calculated for items. The total discrete code learning loss is $\mathcal{L}_{code} = \mathcal{L}_{code}^U + \mathcal{L}_{code}^I$.

### 4.2.3. Reliable and Informative Contrastive View Generation
This is a critical component where CoGCL differentiates itself from existing methods by using discrete codes to generate contrastive views that are both reliable (rooted in observed interactions) and informative (enhancing collaborative structure and semantics).

#### 4.2.3.1. Virtual Neighbor Augmentation via Discrete Codes
To create reliable contrastive views with an enhanced `neighborhood structure`, CoGCL uses discrete codes as `virtual neighbors`. This process aims to alleviate `data sparsity` by expanding the observed interaction graph.

For a given user $u$, a subset of their interacted items $N_u$ (neighbors) is selected with a certain probability $\mathcal{P}$ to form `augmented data` $\mathcal{N}_u^{\mathrm{aug}}$. Two operators are then applied to augment the graph edges:
1.  **"Replace" Operator:** This operator replaces the original interacted items in $\mathcal{N}_u^{\mathrm{aug}}$ with their corresponding discrete codes. The original edges to these items are removed.
2.  **"Add" Operator:** This operator adds the discrete codes of items in $\mathcal{N}_u^{\mathrm{aug}}$ as *additional* virtual neighbors, while retaining the original edges to the items.

    The augmented edges involving user $u$ can be formally expressed as:
$$
\begin{array} { r l } & { { \mathcal E } _ { u } ^ { c } = \left\{ ( u , c _ { i } ^ { h } ) | i \in N _ { u } ^ { \mathrm { a u g } } , h \in \{ 1 , . . . , H \} \right\} , } \\ & { { \mathcal E } _ { u } ^ { r } = \left\{ ( u , i ) | i \in ( N _ { u } \setminus N _ { u } ^ { \mathrm { a u g } } ) \right\} \cup { \mathcal E } _ { u } ^ { c } , } \\ & { { \mathcal E } _ { u } ^ { a } = \{ ( u , i ) | i \in N _ { u } \} \cup { \mathcal E } _ { u } ^ { c } , } \end{array}
$$
Where:
*   $\mathcal{E}_u^c$: The set of new edges created between user $u$ and the discrete codes ($c_i^h$) of the selected items $i \in N_u^{\mathrm{aug}}$. Each item's $H$ discrete codes become virtual neighbors.
*   $\mathcal{E}_u^r$: The set of interaction edges for user $u$ resulting from the "replace" augmentation. It combines original edges to non-augmented items and the new edges to discrete codes $\mathcal{E}_u^c$.
*   $\mathcal{E}_u^a$: The set of interaction edges for user $u$ resulting from the "add" augmentation. It includes all original edges to items $N_u$ and the new edges to discrete codes $\mathcal{E}_u^c$.

    These operations effectively treat discrete codes as `virtual neighbors` of the user. By either replacing original neighbors or adding extra virtual neighbors, the process injects richer neighbor information and helps mitigate graph sparsity. The same augmentation logic is applied symmetrically for items.

To generate a pair of augmented nodes for `Contrastive Learning (CL)`, two rounds of this virtual neighbor augmentation are performed. The resulting augmented graphs are denoted as:
$$
\mathcal { G } ^ { 1 } = ( \widetilde { \mathcal { V } } , \mathcal { E } ^ { o _ { 1 } } ) , \quad \mathcal { G } ^ { 2 } = ( \widetilde { \mathcal { V } } , \mathcal { E } ^ { o _ { 2 } } ) , \quad o _ { 1 } , o _ { 2 } \in \{ r , a \}
$$
Where:
*   $\widetilde{\mathcal{V}} = \{\mathcal{U} \cup C^U \cup \mathcal{I} \cup C^I\}$: The expanded node set, which now includes all users, all items, and all user/item discrete codes. $C^U$ and $C^I$ represent the sets of all discrete codes for users and items, respectively.
*   $o_1, o_2 \in \{r, a\}$: Two stochastic operators chosen from "replace" ($r$) or "add" ($a$). This means $\mathcal{G}^1$ could use "replace" and $\mathcal{G}^2$ could use "add", or both could use the same operator.
*   $\mathcal{E}^{o_1}$ and $\mathcal{E}^{o_2}$: The edge sets generated by applying the chosen virtual neighbor augmentation operations for all users and items across the graph.
    These augmented graphs provide nodes with `abundant` (extensive virtual neighbors) and `homogeneous` (substantial common neighbors via shared codes) structural information. The alignment objective then seeks to bring representations from these two views closer. The discrete codes and augmented graphs are updated once per training epoch.

#### 4.2.3.2. Semantic Relevance Sampling via Discrete Codes
Beyond structure, CoGCL also generates more informative contrastive views by identifying distinct users/items with `similar semantics` as positive pairs. This is more fine-grained than simply treating augmented versions of the same node as positive.

Semantic relevance is assessed in two ways, both leveraging the learned discrete codes:
1.  **Shared Codes:** The discrete codes are designed to correlate with the collaborative semantics of user/item representations. If two users (or items) share a significant number of these codes, it suggests a fine-grained semantic similarity. CoGCL specifically identifies users who share `at least H-1 codes` (where H is the total number of code levels) as semantically positive. This high threshold ensures strong semantic overlap.
2.  **Shared Target:** If two users interact with a common item (a "shared target"), or two items are interacted with by a common user, they are considered `semantically relevant`. This is a form of supervised positive sampling, where the inherent interaction pattern serves as a direct indicator of relevance. This approach has shown effectiveness in other CL contexts like sentence embedding and sequential recommendation.

    From the combined set of instances identified by these two criteria, a positive example $u^+$ is sampled for each user $u$. This forms a more semantically rich positive set for contrastive learning. The same symmetric process applies to items.

### 4.2.4. Triple-View Graph Contrastive Learning
CoGCL integrates the enhanced structural and semantic information through a `triple-view graph contrastive learning` approach. This involves aligning three types of views: two augmented views from virtual neighbor augmentation and one semantically relevant view.

#### 4.2.4.1. Multi-View Representation Encoding
To encode the representations for the augmented graphs, CoGCL introduces `additional learnable embeddings` specifically for the discrete codes. These are denoted as $\mathbf{Z}^c \in \mathbb{R}^{(|C^U| + |C^I|) \times d}$, where $|C^U|$ and $|C^I|$ are the total counts of unique user and item discrete codes respectively.

The initial embedding matrix for the augmented graphs, $\widetilde{\mathbf{Z}}^0$, is formed by concatenating the original ID embeddings $\mathbf{Z}^0$ with these code embeddings $\mathbf{Z}^c$:
$\widetilde{\mathbf{Z}}^0 = [\mathbf{Z}^0 ; \mathbf{Z}^c]$.

The representations for the two augmented views ($\mathcal{G}^1$ and $\mathcal{G}^2$) are then obtained using the same GNN encoder from Section 3.2.1, but with $\widetilde{\mathbf{Z}}^0$ as the initial input and the respective augmented graphs:
$$
\begin{array}{r}
\mathbf{Z}_1^l = \mathrm{GNN}(\rho(\mathbf{Z}_1^{l-1}), \mathcal{G}^1), \quad \mathbf{Z}_2^l = \mathrm{GNN}(\rho(\mathbf{Z}_2^{l-1}), \mathcal{G}^2)
\end{array}
$$
Where:
*   $\mathbf{Z}_1^l$ and $\mathbf{Z}_2^l$: Node representations at layer $l$ for augmented graph $\mathcal{G}^1$ and $\mathcal{G}^2$ respectively.
*   $\rho(\cdot)$: The dropout operation on input representations.
*   The initial representations are set as $\mathbf{Z}_1^0 = \mathbf{Z}_2^0 = \widetilde{\mathbf{Z}}^0$.
    After applying the `Readout` function (as described in Section 3.2.1), the final representations for these two views are denoted as $\mathbf{Z}'$ and $\mathbf{Z}''$.

For the `semantically relevant user/item` view, CoGCL directly uses the node representation obtained from the initial (unaugmented) interaction graph, as described in Section 3.2.1. The dropout applied during the GNN encoding for this view already introduces a form of data augmentation, as different dropout masks lead to different features during two forward propagations, effectively creating distinct views.

#### 4.2.4.2. Alignment Between Neighbor Augmented Views
The two augmented views, $\mathbf{Z}'$ and $\mathbf{Z}''$, derived from `virtual neighbor augmentation`, contain abundant structural information. To leverage this, CoGCL introduces an alignment objective to pull these representations closer. This is done using an `InfoNCE`-like loss. For the user side, the loss is:
$$
\mathcal { L } _ { a u g } ^ { U } = - \left( \log \frac { e ^ { s ( \mathbf { z } _ { u } ^ { \prime } , \mathbf { z } _ { u } ^ { \prime \prime } ) / \tau } } { \sum _ { \tilde { u } \in \mathcal { B } } e ^ { s ( \mathbf { z } _ { u } ^ { \prime } , \mathbf { z } _ { \tilde { u } } ^ { \prime \prime } ) / \tau } } + \log \frac { e ^ { s ( \mathbf { z } _ { u } ^ { \prime \prime } , \mathbf { z } _ { u } ^ { \prime } ) / \tau } } { \sum _ { \tilde { u } \in \mathcal { B } } e ^ { s ( \mathbf { z } _ { u } ^ { \prime \prime } , \mathbf { z } _ { \tilde { u } } ^ { \prime } ) / \tau } } \right)
$$
Where:
*   $\mathcal{L}_{aug}^U$: The augmentation alignment loss for users.
*   $u$: A specific user in the current `batch data` $\mathcal{B}$.
*   $\tilde{u}$: Any user, including $u$ itself, within the batch $\mathcal{B}$.
*   $\mathbf{z}_u'$ and $\mathbf{z}_u''$: The representations of user $u$ from the two different augmented views.
*   $s(\cdot, \cdot)$: Cosine similarity.
*   $\tau$: Temperature coefficient.
    This loss consists of two terms, ensuring `bidirectional alignment`: $\mathbf{z}_u'$ is aligned with $\mathbf{z}_u''$, and vice-versa. Analogously, an item-side loss $\mathcal{L}_{aug}^I$ is computed, and the total augmented view alignment loss is $\mathcal{L}_{aug} = \mathcal{L}_{aug}^U + \mathcal{L}_{aug}^I$.

#### 4.2.4.3. Alignment Between Semantically Relevant Users/Items
To incorporate collaborative `semantic information`, CoGCL aligns users/items with similar collaborative semantics, as identified by `semantic relevance sampling`. For each user $u$, a positive example $u^+$ with similar semantics (based on shared codes or interaction targets) is sampled. The alignment loss, which connects the augmented views of $u$ with its semantically similar counterpart $u^+$, is:
$$
\mathcal { L } _ { s i m } ^ { U } = - \left( \log \frac { e ^ { s ( \mathbf { z } _ { u } ^ { \prime } , \mathbf { z } _ { u ^ { + } } ) / \tau } } { \sum _ { \tilde { u } \in \widetilde { \mathcal { B } } } e ^ { s ( \mathbf { z } _ { u } ^ { \prime } , \mathbf { z } _ { \tilde { u } } ) / \tau } } + \log \frac { e ^ { s ( \mathbf { z } _ { u } ^ { \prime \prime } , \mathbf { z } _ { u ^ { + } } ) / \tau } } { \sum _ { \tilde { u } \in \widetilde { \mathcal { B } } } e ^ { s ( \mathbf { z } _ { u } ^ { \prime \prime } , \mathbf { z } _ { \tilde { u } } ) / \tau } } \right)
$$
Where:
*   $\mathcal{L}_{sim}^U$: The semantic similarity alignment loss for users.
*   $(u, u^+)$: A positive user pair, where $u^+$ is a semantically relevant user for $u$.
*   $\widetilde{\mathcal{B}}$: The sampled data in a batch, which includes $u^+$ and negative samples for $u'$.
*   $\mathbf{z}_u'$ and $\mathbf{z}_u''$: Representations of user $u$ from the two augmented views.
*   $\mathbf{z}_{u^+}$: Representation of the semantically relevant user $u^+$, obtained from the initial GNN encoder (unaugmented graph).
    The two terms in the equation correspond to aligning $\mathbf{z}_u'$ with $\mathbf{z}_{u^+}$ and $\mathbf{z}_u''$ with $\mathbf{z}_{u^+}$, respectively. An analogous item-side loss $\mathcal{L}_{sim}^I$ is computed, and the total semantic alignment loss is $\mathcal{L}_{sim} = \mathcal{L}_{sim}^U + \mathcal{L}_{sim}^I$.

#### 4.2.4.4. Overall Optimization
The entire CoGCL framework is jointly optimized by minimizing a combined loss function that includes the `Bayesian Personalized Ranking (BPR) loss` for recommendation, the `discrete code learning objective`, and the two `contrastive learning losses`.
The `BPR loss` is a standard pairwise ranking loss for implicit feedback data, defined as:
$$
\mathcal{L}_{bpr} = - \sum_{(u, i, j) \in \mathcal{D}} \log \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda_{reg} \|\Theta\|_2^2
$$
Where:
*   $\mathcal{D}$: The set of observed triplets `(u, i, j)`, where user $u$ prefers item $i$ over item $j$.
*   $\sigma(\cdot)$: The sigmoid function.
*   $\hat{y}_{ui}$: The predicted score for user $u$ and item $i$.
*   $\lambda_{reg}$: Regularization coefficient.
*   $\Theta$: All learnable parameters.

    The overall optimization objective for CoGCL is:
$$
\mathcal { L } = \mathcal { L } _ { b p r } + \lambda \mathcal { L } _ { c o d e } + \mu \mathcal { L } _ { a u g } + \eta \mathcal { L } _ { s i m }
$$
Where:
*   $\mathcal{L}_{bpr}$: The primary recommendation loss.
*   $\mathcal{L}_{code}$: The discrete code learning loss (from Section 3.2.2.2).
*   $\mathcal{L}_{aug}$: The alignment loss between neighbor augmented views (from Section 3.4.2).
*   $\mathcal{L}_{sim}$: The alignment loss between semantically relevant users/items (from Section 3.4.3).
*   $\lambda, \mu, \eta$: Hyperparameters that control the trade-off between these different objectives.

## 4.3. Discussion

The paper critically compares CoGCL with existing `graph Contrastive Learning (CL)` methods for `Collaborative Filtering (CF)`, highlighting its unique contributions.

**Comparison with Structural Augmentation Methods:**
*   **Existing Methods (e.g., SGL [51], GFormer [27], LightGCL [4]):** These approaches typically perturb the graph structure (e.g., stochastic node/edge dropout, SVD-based reconstruction). The paper argues that such perturbations, especially on already sparse graphs, can disrupt crucial collaborative information. This leads to `uninformative contrastive views` because key interactions might be lost or the reconstructed graph might not truly capture collaborative semantics.
*   **CoGCL's Differentiation:** CoGCL offers a `reliable and informative` alternative. Instead of perturbing existing structures, it *enhances* them by introducing `discrete codes` as `virtual neighbors`. This process:
    *   **Reliability:** Is strictly based on observed interactions (e.g., a code is associated with an item a user has interacted with).
    *   **Informativeness:** The virtual neighbors (codes) effectively `alleviate data sparsity` by providing more connections and richer local contexts for nodes.
    *   **Benefit:** Aligning representations from two such abundantly structured augmented views is expected to introduce more profound collaborative information into the model.

**Comparison with Representation Augmentation Methods:**
*   **Existing Methods (e.g., SimGCL [60], NCL [29]):** These methods typically involve modeling additional representations (e.g., adding random noise to embeddings, learning hypergraph representations, or cluster centers).
    *   `SimGCL` perturbs embeddings with `random noise`. The paper points out that this `random noise` can interfere with the implicit collaborative semantics in node representations, causing `semantic disruption`.
    *   `NCL` learns cluster centers as anchors, but it's based on the `EM algorithm`, which might not be as `fine-grained` in capturing semantic relevance.
    *   These methods often focus on separating dissimilar instances `indiscriminately`.
*   **CoGCL's Differentiation:** CoGCL's `semantic relevance sampling` offers a more nuanced approach:
    *   **Fine-grained Relevance:** It identifies semantically similar users/items based on `shared discrete codes` (which are learned to embody collaborative semantics) or `shared interaction targets`. This ensures that positive pairs are truly collaboratively relevant.
    *   **Targeted Alignment:** By aligning users/items with explicit semantic relevance, CoGCL aims to `unleash the potential of CL` more effectively, fostering better `semantic learning` within the model. This is in contrast to methods that might just push all non-identical instances apart.

        In summary, CoGCL's novelty lies in its fundamental shift from disruptive perturbations to **constructive and information-rich augmentations** driven by learned discrete codes, leading to more meaningful `positive alignment` and a more potent `self-supervised signal` for recommendation.

# 5. Experimental Setup

## 5.1. Datasets
The experiments evaluate CoGCL on four public datasets that vary in domain, scale, and sparsity, ensuring a comprehensive evaluation.

The following are the statistics from Table 1 of the original paper:

<table>
<thead>
<tr>
<th>Datasets</th>
<th>#Users</th>
<th>#Items</th>
<th>#Interactions</th>
<th>Sparsity</th>
</tr>
</thead>
<tbody>
<tr>
<td>Instrument</td>
<td>48,453</td>
<td>21,413</td>
<td>427,674</td>
<td>99.959%</td>
</tr>
<tr>
<td>Office</td>
<td>181,878</td>
<td>67,409</td>
<td>1,477,820</td>
<td>99.988%</td>
</tr>
<tr>
<td>Gowalla</td>
<td>29,858</td>
<td>40,988</td>
<td>1,027,464</td>
<td>99.916%</td>
</tr>
<tr>
<td>iFashion</td>
<td>300,000</td>
<td>81,614</td>
<td>1,607,813</td>
<td>99.993%</td>
</tr>
</tbody>
</table>

Here's a detailed description of each dataset:
*   **Instrument [17]:** A subset from the `Amazon2023 benchmark` dataset. It represents user-item interactions within the "Musical Instruments" category.
    *   `#Users`: 48,453
    *   `#Items`: 21,413
    *   `#Interactions`: 427,674
    *   `Sparsity`: 99.959% (very sparse)
    *   Preprocessing: Filtered users and items with less than five interactions.
*   **Office [17]:** Another subset from the `Amazon2023 benchmark` dataset, focusing on "Office Products."
    *   `#Users`: 181,878
    *   `#Items`: 67,409
    *   `#Interactions`: 1,477,820
    *   `Sparsity`: 99.988% (extremely sparse)
    *   Preprocessing: Filtered users and items with less than five interactions.
*   **Gowalla [10]:** A location-based social networking dataset, where interactions represent user check-ins at various locations.
    *   `#Users`: 29,858
    *   `#Items`: 40,988
    *   `#Interactions`: 1,027,464
    *   `Sparsity`: 99.916% (sparse)
    *   Preprocessing: A `10-core filtering` is applied, meaning only users and items with at least 10 interactions are retained. This is a common practice to ensure data quality and density.
*   **iFashion [8]:** The `Alibaba-iFashion` dataset, likely related to fashion recommendations.
    *   `#Users`: 300,000
    *   `#Items`: 81,614
    *   `#Interactions`: 1,607,813
    *   `Sparsity`: 99.993% (extremely sparse, among the sparsest)
    *   Preprocessing: Data processed by [51], which involved randomly sampling 300k users and their interactions.

        These datasets were chosen because they represent diverse domains and scales, and importantly, they exhibit varying degrees of `data sparsity`, a key challenge CoGCL aims to address. The preprocessing steps (e.g., filtering low-activity users/items, 10-core filtering) are standard practices in recommendation system research to focus on more active entities and improve data quality.

For each dataset, interactions are split into `training`, `validation`, and `testing` sets with a ratio of 8:1:1.

## 5.2. Evaluation Metrics
The paper uses two widely accepted metrics in recommendation systems to evaluate model performance: `Recall@N` and `Normalized Discounted Cumulative Gain (NDCG)@N`. The value of $N$ is set to 5, 10, and 20. The evaluation is conducted using `full ranking`, meaning predictions are made over the entire item set, not just a subset of sampled negative items, for a more rigorous comparison.

### Recall@N
*   **Conceptual Definition:** `Recall@N` measures the proportion of relevant items that are successfully retrieved within the top $N$ recommendations. It focuses on the ability of a recommendation system to find as many relevant items as possible. A higher Recall@N indicates that the model is effective at identifying a large fraction of the items a user would like among its top $N$ suggestions.
*   **Mathematical Formula:**
    $$
    \text{Recall@N} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\text{Recommended}_{u,N} \cap \text{Relevant}_u|}{|\text{Relevant}_u|}
    $$
*   **Symbol Explanation:**
    *   $|\mathcal{U}|$: The total number of users in the evaluation set.
    *   $u$: A specific user.
    *   $\text{Recommended}_{u,N}$: The set of top $N$ items recommended to user $u$.
    *   $\text{Relevant}_u$: The set of items that are actually relevant to user $u$ (e.g., items the user interacted with in the test set).
    *   $|\cdot|$: Denotes the cardinality (number of elements) of a set.

### Normalized Discounted Cumulative Gain (NDCG)@N
*   **Conceptual Definition:** `NDCG@N` is a metric that evaluates the quality of a ranked list of recommendations. It considers both the `relevance` of the recommended items and their `position` in the list. More relevant items appearing at higher positions (earlier in the list) contribute more to the score. It is normalized by the ideal DCG (IDCG) to ensure scores are comparable across different queries or users. A higher NDCG@N indicates better ranking quality, where the most relevant items are placed at the top.
*   **Mathematical Formula:**
    $$
    \text{NDCG@N} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{\text{DCG@N}_u}{\text{IDCG@N}_u}
    $$
    Where `DCG@N` (Discounted Cumulative Gain) for a user $u$ is calculated as:
    $$
    \text{DCG@N}_u = \sum_{k=1}^{N} \frac{2^{\text{rel}(k)} - 1}{\log_2(k+1)}
    $$
    And `IDCG@N` (Ideal Discounted Cumulative Gain) for a user $u$ is calculated by ranking the relevant items perfectly:
    $$
    \text{IDCG@N}_u = \sum_{k=1}^{|\text{Relevant}_u|, k \le N} \frac{2^{\text{rel}_*(k)} - 1}{\log_2(k+1)}
    $$
*   **Symbol Explanation:**
    *   $|\mathcal{U}|$: The total number of users in the evaluation set.
    *   $u$: A specific user.
    *   $\text{rel}(k)$: The relevance score of the item at position $k$ in the recommended list for user $u$. For implicit feedback, this is typically 1 if the item is relevant and 0 otherwise.
    *   $\text{rel}_*(k)$: The relevance score of the item at position $k$ in the *ideal* ranked list (i.e., relevant items sorted by true relevance, then non-relevant items). For implicit feedback, this is 1 for all relevant items and 0 for non-relevant items, so IDCG is calculated by placing all relevant items at the top $N$ positions.
    *   $\log_2(k+1)$: The discount factor, which reduces the contribution of items appearing further down the list.

## 5.3. Baselines
The paper compares CoGCL against a comprehensive set of competitive baseline models, categorized into `Traditional CF Models` and `CL-based Models`.

### Traditional CF Models:
1.  **BPR [36]:** `Bayesian Personalized Ranking`. A pairwise ranking optimization method for matrix factorization, using `BPR loss` to learn latent representations. It's a fundamental baseline for implicit feedback.
2.  **GCMC [41]:** `Graph Convolutional Matrix Completion`. A GNN-based method that models user-item interactions as a bipartite graph and uses an `auto-encoder framework` for matrix completion.
3.  **NGCF [48]:** `Neural Graph Collaborative Filtering`. Explicitly models high-order connectivity in the user-item interaction graph through message passing, learning expressive user and item embeddings.
4.  **DGCF [49]:** `Disentangled Graph Collaborative Filtering`. Aims to learn disentangled user and item representations, separating different user intents or item aspects to improve recommendation.
5.  **LightGCN [15]:** Simplifies GCN for recommendation by removing non-linear activations and feature transformations, focusing on linear message propagation for efficiency and effectiveness. This often serves as a strong GNN baseline.
6.  **SimpleX [31]:** A straightforward yet robust baseline for collaborative filtering that uses a `cosine contrastive loss` to learn representations, focusing on positive and negative examples.

### CL-based Models:
7.  **SLRec [56]:** `Self-supervised Learning for Large-scale Item Recommendations`. Employs contrastive learning for representation regularization to learn improved latent relationships in large-scale recommendation systems.
8.  **SGL [51]:** `Self-supervised Graph Learning`. Integrates self-supervised learning with graph collaborative filtering. The paper specifically uses `SGL-ED`, which likely refers to `SGL` with `edge dropout` as its augmentation strategy.
9.  **NCL [29]:** `Neighborhood-enriched Contrastive Learning`. Enhances GNN-based recommendation by using contrastive learning with `neighborhood-enriched` information, often involving `cluster centers` as anchors.
10. **HCCF [53]:** `Hypergraph Contrastive Collaborative Filtering`. Uses `hypergraphs` to capture complex, multi-way collaborative relations and integrates this with contrastive learning.
11. **GFormer [27]:** `Graph Transformer for Recommendation`. Leverages a `graph transformer` architecture combined with `masked autoencoding` to distill self-supervised signals and learn invariant collaborative rationales.
12. **SimGCL [60]:** `Simple Graph Contrastive Learning for Recommendation`. A simple yet effective method that generates contrastive views by `adding random noise` to node embeddings, often achieving strong performance.
13. **LightGCL [4]:** `Lightweight Graph Contrastive Learning for Recommendation`. Employs `Singular Value Decomposition (SVD)` to generate augmented views from the adjacency matrix, aiming for a lightweight CL approach.

    These baselines are representative because they cover various foundational recommendation techniques (matrix factorization, graph-based methods) and the latest advancements in self-supervised learning, specifically contrastive learning applied to GNNs. This allows for a thorough comparison of CoGCL's performance against both traditional and cutting-edge approaches.

## 5.4. Implementation Details
The paper outlines specific implementation details to ensure reproducibility and fair comparison:
*   **Optimizer:** `Adam` optimizer is used for model training.
*   **Embedding Dimension:** The embedding dimension ($d$) for all models is uniformly set to 64.
*   **Batch Size:** A batch size of 4096 is used for training.
*   **GNN Layers:** The number of `GNN layers` ($L$) in GNN-based methods (including CoGCL) is set to 3.
*   **Hyperparameter Tuning (Baselines):** For all baseline models, `grid search` is employed to find optimal hyperparameters, guided by the settings reported in their original papers. This ensures that baselines are run under their best possible configurations.
*   **CoGCL Specifics:**
    *   **Discrete Code Method:** `Residual Quantization (RQ)` is used as the default multi-level `vector quantizer`.
    *   **Number of Code Levels ($H$):** Set to 4.
    *   **Temperature Coefficient ($\tau$):** Set to 0.2, used in the `InfoNCE` loss and the discrete code quantization probability calculation.
    *   **Codebook Size ($K$):** Set to 256 for `Instrument` and `Gowalla` datasets. For larger datasets (`Office` and `iFashion`), $K$ is increased to 512 to accommodate their scale.
    *   **Loss Coefficients:**
        *   $\lambda$ (for $\mathcal{L}_{code}$): Tuned in the set {5, 1, 0.5}.
        *   $\mu$ (for $\mathcal{L}_{aug}$): Tuned in {5, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001}.
        *   $\eta$ (for $\mathcal{L}_{sim}$): Tuned in {5, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001}.
    *   **Augmentation Probabilities:** The probabilities for "replace" and "add" operators in `virtual neighbor augmentation` are tuned in {0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6}.
*   **Full Ranking:** All experiments use `full ranking evaluation` over the entire item set, rather than sampling-based evaluation, for robust performance assessment.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results demonstrate that CoGCL consistently achieves the best performance across all four public datasets compared to a wide range of baseline models.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th rowspan="2">Metric</th>
<th>BPR</th>
<th>GCMC</th>
<th>NGCF</th>
<th>DGCF</th>
<th>LightGCN</th>
<th>SimpleX</th>
<th>SLRec</th>
<th>SGL</th>
<th>NCL</th>
<th>HCCF</th>
<th>GFormer</th>
<th>SimGCL</th>
<th>LightGCL</th>
<th rowspan="2">CoGCL</th>
<th rowspan="2">Improv.</th>
</tr>
<tr>
<th colspan="6">Traditional CF Models</th>
<th colspan="7">CL-based Models</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="6">Instrument</td>
<td>Recall@5</td>
<td>0.0293</td>
<td>0.0334</td>
<td>0.0391</td>
<td>0.0401</td>
<td>0.0435</td>
<td>0.0386</td>
<td>0.0381</td>
<td>0.0449</td>
<td>0.0449</td>
<td>0.0456</td>
<td>0.0471</td>
<td>0.0470</td>
<td>0.0468</td>
<td><b>0.0515</b></td>
<td>9.34%</td>
</tr>
<tr>
<td>NDCG@5</td>
<td>0.0194</td>
<td>0.0218</td>
<td>0.0258</td>
<td>0.0269</td>
<td>0.0288</td>
<td>0.0244</td>
<td>0.0256</td>
<td>0.0302</td>
<td>0.0302</td>
<td>0.0303</td>
<td>0.0314</td>
<td>0.0316</td>
<td>0.0310</td>
<td><b>0.0345</b></td>
<td>9.18%</td>
</tr>
<tr>
<td>Recall@10</td>
<td>0.0469</td>
<td>0.0532</td>
<td>0.0617</td>
<td>0.0628</td>
<td>0.0660</td>
<td>0.0631</td>
<td>0.0574</td>
<td>0.0692</td>
<td>0.0685</td>
<td>0.0703</td>
<td>0.0715</td>
<td>0.0717</td>
<td>0.0715</td>
<td><b>0.0788</b></td>
<td>9.90%</td>
</tr>
<tr>
<td>NDCG@10</td>
<td>0.0250</td>
<td>0.0282</td>
<td>0.0331</td>
<td>0.0342</td>
<td>0.0361</td>
<td>0.0324</td>
<td>0.0319</td>
<td>0.0380</td>
<td>0.0377</td>
<td>0.0383</td>
<td>0.0393</td>
<td>0.0395</td>
<td>0.0391</td>
<td><b>0.0435</b></td>
<td>10.13%</td>
</tr>
<tr>
<td>Recall@20</td>
<td>0.0705</td>
<td>0.0824</td>
<td>0.0929</td>
<td>0.0930</td>
<td>0.0979</td>
<td>0.0984</td>
<td>0.0820</td>
<td>0.1026</td>
<td>0.1011</td>
<td>0.1028</td>
<td>0.1041</td>
<td>0.1057</td>
<td>0.1042</td>
<td><b>0.1152</b></td>
<td>8.99%</td>
</tr>
<tr>
<td>NDCG@20</td>
<td>0.0310</td>
<td>0.0357</td>
<td>0.0411</td>
<td>0.0419</td>
<td>0.0442</td>
<td>0.0413</td>
<td>0.0381</td>
<td>0.0466</td>
<td>0.0459</td>
<td>0.0466</td>
<td>0.0478</td>
<td>0.0482</td>
<td>0.0474</td>
<td><b>0.0526</b></td>
<td>9.13%</td>
</tr>
<tr>
<td rowspan="6">Office</td>
<td>Recall@5</td>
<td>0.0204</td>
<td>0.0168</td>
<td>0.0178</td>
<td>0.0258</td>
<td>0.0277</td>
<td>0.0291</td>
<td>0.0294</td>
<td>0.0349</td>
<td>0.0293</td>
<td>0.0340</td>
<td>0.0353</td>
<td>0.0349</td>
<td>0.0338</td>
<td><b>0.0411</b></td>
<td>16.43%</td>
</tr>
<tr>
<td>NDCG@5</td>
<td>0.0144</td>
<td>0.0109</td>
<td>0.0116</td>
<td>0.0177</td>
<td>0.0186</td>
<td>0.0199</td>
<td>0.0209</td>
<td>0.0242</td>
<td>0.0201</td>
<td>0.0230</td>
<td>0.0245</td>
<td>0.0240</td>
<td>0.0232</td>
<td><b>0.0287</b></td>
<td>17.14%</td>
</tr>
<tr>
<td>Recall@10</td>
<td>0.0285</td>
<td>0.0270</td>
<td>0.0279</td>
<td>0.0380</td>
<td>0.0417</td>
<td>0.0422</td>
<td>0.0402</td>
<td>0.0493</td>
<td>0.0434</td>
<td>0.0489</td>
<td>0.0492</td>
<td>0.0494</td>
<td>0.0490</td>
<td><b>0.0582</b></td>
<td>17.81%</td>
</tr>
<tr>
<td>NDCG@10</td>
<td>0.0170</td>
<td>0.0141</td>
<td>0.0149</td>
<td>0.0217</td>
<td>0.0231</td>
<td>0.0241</td>
<td>0.0244</td>
<td>0.0289</td>
<td>0.0243</td>
<td>0.0282</td>
<td>0.0292</td>
<td>0.0289</td>
<td>0.0280</td>
<td><b>0.0343</b></td>
<td>17.47%</td>
</tr>
<tr>
<td>Recall@20</td>
<td>0.0390</td>
<td>0.0410</td>
<td>0.0438</td>
<td>0.0544</td>
<td>0.0605</td>
<td>0.0602</td>
<td>0.0534</td>
<td>0.0681</td>
<td>0.0629</td>
<td>0.0677</td>
<td>0.0672</td>
<td>0.0689</td>
<td>0.0698</td>
<td><b>0.0785</b></td>
<td>12.46%</td>
</tr>
<tr>
<td>NDCG@20</td>
<td>0.0197</td>
<td>0.0178</td>
<td>0.0189</td>
<td>0.0258</td>
<td>0.0279</td>
<td>0.0287</td>
<td>0.0277</td>
<td>0.0336</td>
<td>0.0292</td>
<td>0.0331</td>
<td>0.0338</td>
<td>0.0337</td>
<td>0.0332</td>
<td><b>0.0393</b></td>
<td>14.18%</td>
</tr>
<tr>
<td rowspan="6">Gowalla</td>
<td>Recall@5</td>
<td>0.0781</td>
<td>0.0714</td>
<td>0.0783</td>
<td>0.0895</td>
<td>0.0946</td>
<td>0.0782</td>
<td>0.0689</td>
<td>0.1047</td>
<td>0.1040</td>
<td>0.0836</td>
<td>0.1042</td>
<td>0.1047</td>
<td>0.0947</td>
<td><b>0.1092</b></td>
<td>4.30%</td>
</tr>
<tr>
<td>NDCG@5</td>
<td>0.0707</td>
<td>0.0633</td>
<td>0.0695</td>
<td>0.0801</td>
<td>0.0854</td>
<td>0.0712</td>
<td>0.0613</td>
<td>0.0955</td>
<td>0.0933</td>
<td>0.0749</td>
<td>0.0935</td>
<td>0.0959</td>
<td>0.0860</td>
<td><b>0.0995</b></td>
<td>3.75%</td>
</tr>
<tr>
<td>Recall@10</td>
<td>0.1162</td>
<td>0.1089</td>
<td>0.1150</td>
<td>0.1326</td>
<td>0.1383</td>
<td>0.1187</td>
<td>0.1045</td>
<td>0.1520</td>
<td>0.1508</td>
<td>0.1221</td>
<td>0.1515</td>
<td>0.1525</td>
<td>0.1377</td>
<td><b>0.1592</b></td>
<td>4.39%</td>
</tr>
<tr>
<td>NDCG@10</td>
<td>0.0821</td>
<td>0.0749</td>
<td>0.0808</td>
<td>0.0932</td>
<td>0.0985</td>
<td>0.0834</td>
<td>0.0722</td>
<td>0.1092</td>
<td>0.1078</td>
<td>0.0866</td>
<td>0.1085</td>
<td>0.1100</td>
<td>0.0988</td>
<td><b>0.1145</b></td>
<td>4.09%</td>
</tr>
<tr>
<td>Recall@20</td>
<td>0.1695</td>
<td>0.1626</td>
<td>0.1666</td>
<td>0.1914</td>
<td>0.2002</td>
<td>0.1756</td>
<td>0.1552</td>
<td>0.2160</td>
<td>0.2130</td>
<td>0.1794</td>
<td>0.2166</td>
<td>0.2181</td>
<td>0.1978</td>
<td><b>0.2253</b></td>
<td>3.30%</td>
</tr>
<tr>
<td>NDCG@20</td>
<td>0.0973</td>
<td>0.0903</td>
<td>0.0956</td>
<td>0.1100</td>
<td>0.1161</td>
<td>0.0996</td>
<td>0.0868</td>
<td>0.1274</td>
<td>0.1254</td>
<td>0.1029</td>
<td>0.1271</td>
<td>0.1286</td>
<td>0.1159</td>
<td><b>0.1333</b></td>
<td>3.65%</td>
</tr>
<tr>
<td rowspan="6">iFashion</td>
<td>Recall@5</td>
<td>0.0195</td>
<td>0.0240</td>
<td>0.0234</td>
<td>0.0297</td>
<td>0.0309</td>
<td>0.0345</td>
<td>0.0237</td>
<td>0.0377</td>
<td>0.0330</td>
<td>0.0419</td>
<td>0.0354</td>
<td>0.0401</td>
<td>0.0423</td>
<td><b>0.0463</b></td>
<td>9.46%</td>
</tr>
<tr>
<td>NDCG@5</td>
<td>0.0128</td>
<td>0.0156</td>
<td>0.0151</td>
<td>0.0197</td>
<td>0.0205</td>
<td>0.0231</td>
<td>0.0157</td>
<td>0.0252</td>
<td>0.0219</td>
<td>0.0280</td>
<td>0.0235</td>
<td>0.0267</td>
<td>0.0284</td>
<td><b>0.0310</b></td>
<td>9.15%</td>
</tr>
<tr>
<td>Recall@10</td>
<td>0.0307</td>
<td>0.0393</td>
<td>0.0384</td>
<td>0.0459</td>
<td>0.0481</td>
<td>0.0525</td>
<td>0.0361</td>
<td>0.0574</td>
<td>0.0501</td>
<td>0.0636</td>
<td>0.0540</td>
<td>0.0608</td>
<td>0.0641</td>
<td><b>0.0696</b></td>
<td>8.58%</td>
</tr>
<tr>
<td>NDCG@10</td>
<td>0.0164</td>
<td>0.0206</td>
<td>0.0199</td>
<td>0.0249</td>
<td>0.0260</td>
<td>0.0289</td>
<td>0.0198</td>
<td>0.0315</td>
<td>0.0274</td>
<td>0.0350</td>
<td>0.0294</td>
<td>0.0334</td>
<td>0.0354</td>
<td><b>0.0386</b></td>
<td>9.04%</td>
</tr>
<tr>
<td>Recall@20</td>
<td>0.0470</td>
<td>0.0623</td>
<td>0.0608</td>
<td>0.0685</td>
<td>0.0716</td>
<td>0.0770</td>
<td>0.0535</td>
<td>0.0846</td>
<td>0.0742</td>
<td>0.0929</td>
<td>0.0790</td>
<td>0.0897</td>
<td>0.0932</td>
<td><b>0.1010</b></td>
<td>8.37%</td>
</tr>
<tr>
<td>NDCG@20</td>
<td>0.0206</td>
<td>0.0264</td>
<td>0.0256</td>
<td>0.0307</td>
<td>0.0320</td>
<td>0.0351</td>
<td>0.0242</td>
<td>0.0384</td>
<td>0.0335</td>
<td>0.0425</td>
<td>0.0358</td>
<td>0.0407</td>
<td>0.0428</td>
<td><b>0.0465</b></td>
<td>8.64%</td>
</tr>
</tbody>
</table>

**Key Observations from Overall Performance:**
1.  **CL-based Methods Outperform Traditional Methods:** Generally, `contrastive learning (CL)`-based methods (SGL, NCL, SimGCL, LightGCL, etc.) show superior performance compared to `traditional matrix factorization (MF)` methods (BPR, SimpleX) and `GNN-only` methods (NGCF, LightGCN). This confirms the value of `self-supervised signals` in alleviating data sparsity and enhancing representation learning for recommendation.
2.  **Varied Strengths of CL-based Methods:**
    *   `SimGCL` (a representation augmentation method) performs best among baselines on `Instrument` and `Gowalla`, suggesting that `random noise` for uniformity is effective in some contexts.
    *   `GFormer` and `LightGCL` (structure augmentation methods) are more competitive on `Office` and `iFashion` respectively, implying that carefully designed structural augmentations can be beneficial.
    *   `SGL` (stochastic edge/node dropout) sometimes underperforms, reinforcing the paper's argument that naive structural perturbations can disrupt crucial information.
3.  **CoGCL's Consistent Superiority:** CoGCL consistently achieves the **highest scores** across all datasets and all metrics (`Recall@N`, `NDCG@N`). The improvement percentages range from 3.30% to 17.81% over the best baseline, highlighting its robustness and effectiveness.
    *   The largest improvements are seen on the `Office` dataset (e.g., 17.81% in Recall@10, 17.47% in NDCG@10), an extremely sparse dataset (99.988% sparsity). This suggests CoGCL is particularly effective in highly sparse environments.
    *   The improvements are attributed to CoGCL's ability to construct contrastive views with stronger collaborative information through `discrete codes`, `virtual neighbor augmentation`, and `semantic relevance sampling`, which collectively provide richer self-supervised signals.

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Ablation Study of Data Augmentation
This study investigates the contribution of each specific data augmentation method within CoGCL.
The following chart (Figure 4 from the original paper) shows the impact of different data augmentation methods on NDCG@10 for Instrument and Office datasets:

![Figure 4: Ablation study of data augmentation methods.](images/4.jpg)
*该图像是图表，展示了不同数据增强方法对 NDCG@10 的影响，左侧为 Instrument 数据集，右侧为 Office 数据集。图中包含 'w/o Replace'、'w/o Add'、'w/o Shared-C'、'w/o Shared-T' 和 'CoGCL' 的对比结果。*

Alt text: Figure 4: Ablation study of data augmentation methods.
*   **`w/o Replace`:** Removes the "replace" operator in `virtual neighbor augmentation`.
*   **`w/o Add`:** Removes the "add" operator in `virtual neighbor augmentation`.
*   **`w/o Shared-C`:** Removes `semantic relevance sampling` based on `shared codes`.
*   **`w/o Shared-T`:** Removes `semantic relevance sampling` based on `shared interaction targets`.

**Analysis:**
The results in Figure 4 consistently show that removing any of these data augmentation components leads to a decrease in performance across both `Instrument` and `Office` datasets. This empirically validates that:
*   Both "replace" and "add" strategies in `virtual neighbor augmentation` contribute positively to the model's effectiveness, enriching the neighborhood structure.
*   Both `shared codes` and `shared targets` are important for identifying `semantic relevance` and generating informative positive pairs.
    The decline in performance confirms that all designed data augmentation methods within CoGCL are useful and contribute to the overall performance improvement, suggesting they successfully introduce stronger collaborative information without disruption.

### 6.2.2. Ablation Study of Triple-View Graph Contrastive Learning
This study delves into the roles of `alignment` and `uniformity` within CoGCL's contrastive learning objectives ($\mathcal{L}_{aug}$ and $\mathcal{L}_{sim}$). The paper uses variants where the gradient for either alignment or uniformity is stopped.

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th colspan="2">Instrument</th>
<th colspan="2">Office</th>
</tr>
<tr>
<th>Recall@10</th>
<th>NDCG@10</th>
<th>Recall@10</th>
<th>NDCG@10</th>
</tr>
</thead>
<tbody>
<tr>
<td>LightGCN</td>
<td>0.0660</td>
<td>0.0361</td>
<td>0.0417</td>
<td>0.0231</td>
</tr>
<tr>
<td>CoGCL</td>
<td>0.0788</td>
<td>0.0435</td>
<td>0.0582</td>
<td>0.0343</td>
</tr>
<tr>
<td>w/o A</td>
<td>0.0726</td>
<td>0.0401</td>
<td>0.0490</td>
<td>0.0280</td>
</tr>
<tr>
<td>w/o U</td>
<td>0.0703</td>
<td>0.0384</td>
<td>0.0465</td>
<td>0.0267</td>
</tr>
<tr>
<td>w/o AA</td>
<td>0.0741</td>
<td>0.0411</td>
<td>0.0536</td>
<td>0.0315</td>
</tr>
<tr>
<td>w/o AU</td>
<td>0.0762</td>
<td>0.0421</td>
<td>0.0542</td>
<td>0.0306</td>
</tr>
<tr>
<td>w/o SA</td>
<td>0.0767</td>
<td>0.0422</td>
<td>0.0554</td>
<td>0.0329</td>
</tr>
<tr>
<td>w/o SU</td>
<td>0.0779</td>
<td>0.0429</td>
<td>0.0574</td>
<td>0.0336</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Overall Impact of CL Components:** Compared to `LightGCN`, CoGCL shows substantial improvements, demonstrating the effectiveness of the entire `triple-view graph contrastive learning` approach.
*   **Importance of Alignment and Uniformity:**
    *   `w/o A` (disabling alignment in both $\mathcal{L}_{aug}$ and $\mathcal{L}_{sim}$): Leads to a significant performance drop (e.g., NDCG@10 drops from 0.0435 to 0.0401 on Instrument).
    *   `w/o U` (disabling uniformity in both $\mathcal{L}_{aug}$ and $\mathcal{L}_{sim}$): Also causes a notable performance degradation (e.g., NDCG@10 drops from 0.0435 to 0.0384 on Instrument).
        These results indicate that both `alignment` (pulling positives together) and `uniformity` (spreading representations out) are crucial for CoGCL's effectiveness, unlike some previous findings where uniformity was primarily responsible for gains (as discussed in Section 2.2).
*   **Individual Contributions of Alignment:**
    *   `w/o AA` (disabling alignment for $\mathcal{L}_{aug}$ only): Shows a pronounced decrease in performance.
    *   `w/o SA` (disabling alignment for $\mathcal{L}_{sim}$ only): Also incurs a noticeable drop.
        This provides strong evidence that the explicit `alignment` between the two types of positives (neighbor-augmented views and semantically relevant users/items) successfully introduces enhanced collaborative information into the model.
*   **Individual Contributions of Uniformity:**
    *   `w/o AU` (disabling uniformity for $\mathcal{L}_{aug}$ only) and `w/o SU` (disabling uniformity for $\mathcal{L}_{sim}$ only) lead to smaller performance drops compared to disabling alignment. The paper suggests this might be due to a `shared uniformity effect` between the two CL losses, where they mutually reinforce each other, meaning that some uniformity is still implicitly maintained even if one is explicitly disabled.

        Overall, the ablation study confirms that CoGCL's strength comes from a balanced interplay of alignment and uniformity, with its `collaborative information-aware` alignment being a particularly strong driver of performance.

### 6.2.3. Performance Comparison w.r.t. Different Discrete Code Learning Methods
This analysis compares CoGCL's proposed end-to-end discrete code learning method with several alternatives.
The following chart (Figure 5 from the original paper) shows the performance comparison of different discrete code learning methods:

![Figure 5: Performance comparison of different discrete code learning methods.](images/5.jpg)
*该图像是图表，展示了不同离散编码学习方法在Instrument和Office数据集上的性能比较，包括Recall@10和NDCG@10指标。使用不同编码方法的结果显示，CoGCL方法在多个指标上表现优异。*

Alt text: Figure 5: Performance comparison of different discrete code learning methods.
*   **`Non-Learnable Code`:** Uses the `Faiss library` to generate discrete codes from pre-trained `LightGCN` embeddings. These codes are fixed during training.
*   **`Euclidean Code`:** Uses `Euclidean distance` instead of `cosine similarity` in the quantization step (Eq. 6), which is common in traditional `RQ`.
*   **`PQ Code`:** Employs `Product Quantization (PQ)` instead of `Residual Quantization (RQ)` as the multi-level quantizer.
*   **`CoGCL`:** Uses the proposed end-to-end `RQ` with `cosine similarity`.

**Analysis:**
*   **Importance of Learnable Codes:** `Non-Learnable Code` performs worse than CoGCL. This highlights the importance of learning discrete codes `end-to-end` alongside the recommendation task. Allowing the codes to adapt and improve collaboratively with the GNN representations ensures they remain informative and reliable throughout training. Fixed codes might not capture the evolving collaborative semantics.
*   **Cosine vs. Euclidean Similarity:** `Euclidean Code` performs worse than CoGCL. This validates the design choice of using `cosine similarity` for quantization. By synchronizing the similarity measure in VQ with that used in `InfoNCE` loss, the discrete codes become more aligned with the geometric properties (directions) of the embedding space relevant for contrastive learning.
*   **RQ vs. PQ:** `PQ Code` also underperforms CoGCL. This suggests that `Residual Quantization (RQ)` is more suitable for CoGCL's objectives than `Product Quantization (PQ)`. RQ establishes `conditional probability relationships` between codes at different levels (quantizing residuals), allowing for a more granular and refined semantic modeling than PQ, which treats sub-vectors (and thus their codes) independently.

    These results confirm the advancedness and effectiveness of CoGCL's specific approach to end-to-end discrete code learning, particularly the use of `RQ` and `cosine similarity`.

### 6.2.4. Performance Comparison w.r.t. Data Sparsity
This study examines CoGCL's robustness and effectiveness across different levels of `data sparsity`. Users are divided into five groups based on their number of interactions, with each group having the same number of users.
The following chart (Figure 6 from the original paper) shows the performance comparison on user groups with different sparsity levels:

![Figure 6: Performance comparison on user groups with different sparsity levels.](images/6.jpg)
*该图像是一个条形图，展示了在用户组不同稀疏程度下，SimGCL和CoGCL的NDCG@10性能比较。左侧为Instrument数据集，右侧为Office数据集。图中可以看出，CoGCL在各稀疏程度下的表现优于SimGCL。*

Alt text: Figure 6: Performance comparison on user groups with different sparsity levels.
*   The x-axis represents user groups from `least sparse` to `most sparse` (Group 1 to Group 5).
*   The y-axis represents `NDCG@10`.
*   `SimGCL` is used as a strong baseline for comparison.

**Analysis:**
*   **CoGCL's Consistent Outperformance:** CoGCL consistently outperforms `SimGCL` across all sparsity levels on both `Instrument` and `Office` datasets.
*   **Superiority in High Sparsity:** The performance gap between CoGCL and `SimGCL` appears to be most significant for the `highly sparse user groups` (Groups 4 and 5). For example, on the `Office` dataset, CoGCL maintains a much higher NDCG@10 for the sparsest users.
    This phenomenon indicates that CoGCL is particularly effective in scenarios with `sparse interactions`. The ability to introduce `additional insights` through its `collaborative information-aware contrastive views` (via `virtual neighbor augmentation` and `semantic relevance sampling`) allows it to learn high-quality representations even when direct interaction data is scarce. This directly addresses one of the core motivations of the paper.

### 6.2.5. Hyperparameter Tuning
#### 6.2.5.1. CL loss coefficients $\mu$ and $\eta$
The following chart (Figure 7 from the original paper) shows the performance comparison of different CL loss coefficients:

![Figure 7: Performance comparison of different CL loss coefficients.](images/7.jpg)
*该图像是图表，展示了不同 CL 损失系数 `oldsymbol{eta}` 对 Instrument 和 Office 数据集的 Recall@10 和 NDCG@10 性能的影响。图中分别展示了不同参数下的回忆率和归一化折扣累积增益的变化趋势。*

Alt text: Figure 7: Performance comparison of different CL loss coefficients.
*   The chart shows the effect of varying $\mu$ (weight for $\mathcal{L}_{aug}$) and $\eta$ (weight for $\mathcal{L}_{sim}$) on `Recall@10` and `NDCG@10` for `Instrument` and `Office` datasets.

**Analysis:**
*   **Sensitivity to $\mu$:** The performance is sensitive to $\mu$. Both `too large` and `too small` values of $\mu$ lead to suboptimal performance. This indicates that the `alignment between augmented views` needs to be carefully balanced; an overly strong or weak signal can degrade overall results. For `Instrument`, optimal $\mu$ is 0.1; for `Office`, it's 1.0.
*   **Sensitivity to $\eta$:** An excessively large value of $\eta$ causes a `sharp drop` in performance. This suggests that the `alignment with semantically relevant users/items` is very powerful but can be detrimental if overemphasized, potentially leading to `over-clustering` or `representation collapse` if not properly balanced. Optimal $\eta$ values are 0.02 for `Instrument` and 0.2 for `Office`.
*   **Relative Magnitude:** The optimal value of $\eta$ is generally `smaller` than that of $\mu$. This implies that while `semantic relevance` is crucial, its influence needs to be more carefully constrained compared to the structural augmentation.
    These findings emphasize the importance of careful hyperparameter tuning to achieve the optimal balance between the `recommendation loss`, `discrete code learning`, `structural augmentation alignment`, and `semantic relevance alignment`.

#### 6.2.5.2. Augmentation probabilities
The following chart (Figure 8 from the original paper) shows the performance comparison of different augmentation probabilities:

![Figure 8: Performance comparison of different augmentation probabilities.](images/8.jpg)
*该图像是图表，展示了不同增强概率下的性能比较，包含两种情况：替代（replace）和添加（add）。图表中展示了两个数据集（Instrument 和 Office）的 Recall@10 和 NDCG@10 的变化情况，左侧为概率 'replace'，右侧为概率 'add'。*

Alt text: Figure 8: Performance comparison of different augmentation probabilities.
*   The chart shows the effect of varying probabilities for the "replace" and "add" operators in `virtual neighbor augmentation` on `Recall@10` and `NDCG@10` for `Instrument` and `Office` datasets.

**Analysis:**
*   **Optimal Range:** The performance degrades if the probabilities for either "replace" or "add" are `excessively high` or `too low`. This suggests there is an optimal range where the right amount of virtual neighbor augmentation is beneficial.
*   **Dataset Specificity:**
    *   For `Instrument`, the ideal probability for "replace" is 0.3, and for "add" is 0.2.
    *   For `Office`, the optimal probability for "replace" is 0.2, and for "add" is 0.5.
        This indicates that the optimal `augmentation strategy` can be `dataset-dependent`, likely reflecting differences in graph density and structure. A moderate level of augmentation is generally preferred to introduce richer context without overwhelming or distorting the original collaborative signals.

### 6.2.6. Embedding Distribution w.r.t. Augmentation Ratio
This analysis uses `t-SNE` [43] and `Gaussian Kernel Density Estimation (KDE)` [3] to visualize the user embedding distributions under different augmentation ratios, providing an intuitive understanding of CoGCL's learned representations.
The following chart (Figure 9 from the original paper) shows the embedding distribution of different data augmentation ratios on Instrument dataset:

![Figure 9: Embedding distribution of different data augmentation ratios on Instrument dataset. The transition from green to blue signifies a gradual increase in embedding density.](images/9.jpg)
*该图像是一个图表，展示了不同数据增强比例下的嵌入分布，包括LightGCN、CoGCL 2p、CoGCL、CoGCL 0.5p和SimGCL五种方法。颜色从绿色渐变为蓝色，表示嵌入密度的逐渐增加。*

Alt text: Figure 9: Embedding distribution of different data augmentation ratios on Instrument dataset. The transition from green to blue signifies a gradual increase in embedding density.
*   `LightGCN`: Baseline without CL.
*   `SimGCL`: CL baseline with random noise.
*   `CoGCL`: Proposed method at optimal augmentation ratio.
*   `CoGCL 2p`: CoGCL with probabilities (for both "replace" and "add") set to twice the optimal values.
*   `CoGCL 0.5p`: CoGCL with probabilities set to half the optimal values.

**Analysis:**
*   **Uniformity by CL:** Both `CoGCL` and `SimGCL` exhibit `more uniform embedding distributions` compared to `LightGCN`. This confirms that `contrastive learning` generally helps to spread out representations, preventing `representation collapse` and improving `uniformity`.
*   **CoGCL's Balance:** CoGCL achieves a good `trade-off between clustering and uniformity`. While it maintains uniformity, it also allows for meaningful clusters, suggesting it captures underlying collaborative semantics effectively. In contrast, `SimGCL` might lean more heavily on uniformity through random noise, potentially at the cost of some semantic coherence.
*   **Effect of Augmentation Ratio:** As the augmentation ratio `rises` (from `CoGCL 0.5p` to `CoGCL` to `CoGCL 2p`), the embeddings tend to exhibit a `more clustered pattern`. This indicates that `higher augmentation probabilities` (i.e., more virtual neighbors or more semantic positives) lead to a stronger tendency for `clustering` of similar users/items. This is likely because increased augmentation provides more positive signals, pushing related entities closer. However, too much clustering (as implied by the suboptimal performance at `2p` in the hyperparameter tuning) can be detrimental, suggesting that `over-augmentation` might merge distinct semantic groups.

    This visualization provides qualitative evidence that CoGCL learns high-quality, well-structured representations by effectively balancing the desire for uniform distribution with the need to cluster semantically similar entities, thanks to its reliable and informative augmentation strategies.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper introduces `CoGCL`, a novel framework aimed at enhancing `graph contrastive learning (CL)` for `collaborative filtering (CF)`. The central idea is to generate `reliable and informative contrastive views` by leveraging `discrete codes` that are rich in collaborative information.

The key contributions and findings are:
1.  **End-to-End Discrete Code Learning:** A `multi-level vector quantizer` is integrated into the training process to map user and item representations into discrete codes, specifically designed to capture fine-grained collaborative semantics.
2.  **Enhanced Contrastive View Generation:** These discrete codes are used to:
    *   Perform `virtual neighbor augmentation`, treating codes as virtual neighbors to enrich graph structure and `alleviate data sparsity`.
    *   Enable `semantic relevance sampling` by identifying users/items sharing codes or interaction targets, creating more meaningful positive pairs.
3.  **Triple-View Graph Contrastive Learning:** A novel objective function is proposed to align representations from two augmented structural views and one semantically relevant view, effectively integrating diverse collaborative signals.
    Extensive experiments across four public datasets demonstrate CoGCL's superior performance compared to state-of-the-art baselines. Further analyses confirm that all proposed components contribute positively, particularly highlighting CoGCL's robustness in `sparse data scenarios` and its ability to achieve a better balance between `alignment` and `uniformity` in the learned embeddings.

## 7.2. Limitations & Future Work
The authors explicitly mention that a primary direction for future work is to improve the `scalability` of CoGCL. This suggests that, while effective, the current framework might have computational bottlenecks that limit its application to extremely large-scale industrial scenarios or different recommendation tasks.

Specifically, they aim to extend CoGCL to:
*   **Click-Through Rate (CTR) prediction:** This task often involves rich feature sets beyond just user-item interactions and requires models to predict the probability of a click. Adapting CoGCL to handle auxiliary information and a different prediction objective would be a significant extension.
*   **Sequential recommendation:** This task focuses on predicting the next item a user will interact with, based on their historical sequence of interactions. Integrating the discrete codes and augmented views into sequential models would require adapting the graph construction and contrastive learning objectives to sequence dynamics.

    Potential limitations implied by the paper's focus and future work include:
*   **Computational Overhead:** Learning and managing multi-level discrete codebooks ($H \times K \times d$ parameters for codes, plus the additional graph propagation for augmented views) could introduce significant computational costs during training, especially with larger $K$ or $H$. The scalability issue likely stems from this.
*   **Hyperparameter Sensitivity:** The ablation studies show that the model is sensitive to the `loss coefficients (`\mu, \eta`)` and `augmentation probabilities`. Optimal performance requires careful tuning, which can be resource-intensive.
*   **Code Cold-Start:** While discrete codes alleviate data sparsity for existing users/items, the paper doesn't explicitly discuss how new users or items (with no interactions, thus no codes) are handled in the `discrete code learning` phase.

## 7.3. Personal Insights & Critique
This paper presents a compelling and well-justified approach to enhancing graph contrastive learning for recommendation. The core insight—that arbitrary perturbations can harm collaborative information—is crucial, and the solution using learned discrete codes is elegant.

**Strengths:**
*   **Strong Motivation:** The empirical analysis in Section 2.2, demonstrating the ineffectiveness of alignment in existing perturbation-based CL, effectively sets the stage for the proposed method.
*   **Principled Augmentation:** Moving beyond random perturbations to `collaborative information-aware` augmentation (virtual neighbors, semantic relevance) is a significant conceptual advancement. It makes the self-supervised signals more meaningful.
*   **Multi-Level Discrete Codes:** The use of `multi-level vector quantization` allows for a fine-grained, learnable discretization of user/item semantics, which is a powerful way to capture and share collaborative knowledge. The choice of `cosine similarity` in VQ aligns well with the downstream CL objectives.
*   **Comprehensive Evaluation:** The experiments are thorough, covering multiple datasets, a wide range of baselines, and detailed ablation studies.

**Potential Issues/Areas for Improvement:**
*   **Computational Efficiency:** While the paper mentions that $H \times K \ll |\mathcal{U}|$ and $H \times K \ll |\mathcal{I}|$, which keeps the codebook parameters manageable, the process of constructing augmented graphs with codes as virtual nodes and running GNNs on these expanded graphs could still be computationally demanding during training. Future work on scalability is indeed critical.
*   **Interpretability of Discrete Codes:** While codes are "rich in collaborative information," the paper doesn't delve deeply into what specific collaborative patterns or semantic clusters these codes represent. Further analysis of the codebook embeddings or clusters could offer more interpretability.
*   **Generalizability of "H-1 Shared Codes" rule:** The rule of "at least H-1 codes" for semantic relevance sampling is an empirical heuristic. While it worked well, exploring adaptive or more sophisticated ways to determine semantic similarity based on codes could be an interesting avenue.
*   **Dynamic Codebook:** The codebook is updated once per epoch. Investigating more dynamic or online updating strategies for codebooks, especially in highly dynamic recommendation environments, might yield further benefits.

**Transferability and Broader Impact:**
The methodology of using `learnable discrete codes` for `reliable and informative data augmentation` has broad transferability.
*   **Other Graph-based Tasks:** This approach could be applied to other graph-based tasks beyond recommendation where `graph augmentation` is used, but preserving underlying semantic relationships is critical (e.g., node classification, link prediction in knowledge graphs).
*   **Sequential Recommendation:** As mentioned by the authors, discrete codes could simplify item representation in `sequential models`, allowing for more efficient modeling of long sequences and capturing user interests at different levels of granularity.
*   **LLM-based Recommendation:** The concept of discrete codes is already gaining traction in `Large Language Model (LLM)`-based recommendation for indexing and representing items. CoGCL's method of learning these codes to embed collaborative semantics could further enhance LLM capabilities by providing more structured and semantically rich item representations.
*   **Fairness and Explainability:** Discrete codes might offer new avenues for `explainable recommendation` by mapping user/item properties to interpretable code combinations. They could also potentially be designed to encode diverse preferences, contributing to `fairness` by ensuring broader representation.

    Overall, CoGCL offers a significant step forward in making `contrastive learning` a more powerful and trustworthy tool for `recommender systems`, particularly by addressing the often-overlooked problem of maintaining collaborative information integrity during view generation.